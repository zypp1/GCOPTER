#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;
    ros::Subscriber planReqSub;
    ros::Publisher  exprtPub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> traj;
    double trajStamp;

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());

        planReqSub = nh.subscribe("/nmp/plan_request", 1, &GlobalPlanner::planReqCallBack, this,
                                  ros::TransportHints().tcpNoDelay());
        exprtPub = nh.advertise<std_msgs::Float64MultiArray>("/nmp/expert_traj", 1);
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                   startGoal[1],
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            std::vector<Eigen::MatrixX4d> hPolys;
            std::vector<Eigen::Vector3d> pc;
            voxelMap.getSurf(pc);

            sfc_gen::convexCover(route,
                                 pc,
                                 voxelMap.getOrigin(),
                                 voxelMap.getCorner(),
                                 7.0,
                                 3.0,
                                 hPolys);
            sfc_gen::shortCut(hPolys);

            if (route.size() > 1)
            {
                visualizer.visualizePolytope(hPolys);

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC gcopter;

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                traj.clear();

                if (!gcopter.setup(config.weightT,
                                   iniState, finState,
                                   hPolys, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
                {
                    return;
                }

                if (traj.getPieceNum() > 0)
                {
                    trajStamp = ros::Time::now().toSec();
                    visualizer.visualize(traj, route);
                }
            }
        }
    }

    // NMP data collection: programmatic plan request + trajectory publish.
    // Mirrors plan() but takes start/goal + start velocity from the request and
    // serializes the optimized MINCO trajectory onto /nmp/expert_traj.
    inline void planReqCallBack(const std_msgs::Float64MultiArray::ConstPtr &msg)
    {
        std_msgs::Float64MultiArray out;
        if (!mapInitialized || msg->data.size() < 10)
        {
            out.data.push_back(0.0);          // not ready / bad request
            exprtPub.publish(out);
            return;
        }
        const Eigen::Vector3d s(msg->data[0], msg->data[1], msg->data[2]);
        const Eigen::Vector3d sv(msg->data[4], msg->data[5], msg->data[6]);
        const Eigen::Vector3d g(msg->data[7], msg->data[8], msg->data[9]);

        std::vector<Eigen::Vector3d> route;
        try                               // informed RRT* (OMPL) throws on degenerate
        {                                 // geometry (e.g. start==goal PHS); catch it
            sfc_gen::planPath<voxel_map::VoxelMap>(s, g,
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
        }
        catch (const std::exception &e)
        {
            ROS_WARN("planReqCallBack: planPath threw (%s) -> failure", e.what());
            out.data.push_back(0.0);
            exprtPub.publish(out);
            return;
        }
        if (route.size() <= 1)            // start/goal in obstacle or unreachable:
        {                                 // convexCover() segfaults on a degenerate route
            out.data.push_back(0.0);
            exprtPub.publish(out);
            return;
        }
        std::vector<Eigen::MatrixX4d> hPolys;
        std::vector<Eigen::Vector3d> pc;
        voxelMap.getSurf(pc);
        sfc_gen::convexCover(route, pc,
                             voxelMap.getOrigin(), voxelMap.getCorner(),
                             7.0, 3.0, hPolys);
        sfc_gen::shortCut(hPolys);

        if (route.size() > 1)
        {
            Eigen::Matrix3d iniState;
            Eigen::Matrix3d finState;
            iniState << route.front(), sv, Eigen::Vector3d::Zero();   // start velocity from request
            finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

            gcopter::GCOPTER_PolytopeSFC gcopter;
            Eigen::VectorXd magnitudeBounds(5);
            Eigen::VectorXd penaltyWeights(5);
            Eigen::VectorXd physicalParams(6);
            magnitudeBounds(0) = config.maxVelMag;
            magnitudeBounds(1) = config.maxBdrMag;
            magnitudeBounds(2) = config.maxTiltAngle;
            magnitudeBounds(3) = config.minThrust;
            magnitudeBounds(4) = config.maxThrust;
            penaltyWeights(0) = (config.chiVec)[0];
            penaltyWeights(1) = (config.chiVec)[1];
            penaltyWeights(2) = (config.chiVec)[2];
            penaltyWeights(3) = (config.chiVec)[3];
            penaltyWeights(4) = (config.chiVec)[4];
            physicalParams(0) = config.vehicleMass;
            physicalParams(1) = config.gravAcc;
            physicalParams(2) = config.horizDrag;
            physicalParams(3) = config.vertDrag;
            physicalParams(4) = config.parasDrag;
            physicalParams(5) = config.speedEps;
            const int quadratureRes = config.integralIntervs;

            traj.clear();
            if (gcopter.setup(config.weightT, iniState, finState, hPolys, INFINITY,
                              config.smoothingEps, quadratureRes,
                              magnitudeBounds, penaltyWeights, physicalParams) &&
                !std::isinf(gcopter.optimize(traj, config.relCostTol)) &&
                traj.getPieceNum() > 0)
            {
                trajStamp = ros::Time::now().toSec();
                visualizer.visualize(traj, route);

                const int N = traj.getPieceNum();
                const Eigen::VectorXd T = traj.getDurations();
                const double tot = traj.getTotalDuration();
                const double dt = 0.05;
                const int K = (int)std::floor(tot / dt) + 1;
                out.data.push_back(1.0);                 // success
                out.data.push_back((double)N);
                for (int i = 0; i < N; ++i) out.data.push_back(T(i));
                out.data.push_back((double)K);
                for (int k = 0; k < K; ++k)
                {
                    const Eigen::Vector3d x = traj.getPos(std::min(k * dt, tot));
                    out.data.push_back(x.x());
                    out.data.push_back(x.y());
                    out.data.push_back(x.z());
                }
                exprtPub.publish(out);
                return;
            }
        }
        out.data.push_back(0.0);                          // planning failed
        exprtPub.publish(out);
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0)
        {
            const double delta = ros::Time::now().toSec() - trajStamp;
            if (delta > 0.0 && delta < traj.getTotalDuration())
            {
                double thr;
                Eigen::Vector4d quat;
                Eigen::Vector3d omg;

                flatmap.forward(traj.getVel(delta),
                                traj.getAcc(delta),
                                traj.getJer(delta),
                                0.0, 0.0,
                                thr, quat, omg);
                double speed = traj.getVel(delta).norm();
                double bodyratemag = omg.norm();
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;
                visualizer.speedPub.publish(speedMsg);
                visualizer.thrPub.publish(thrMsg);
                visualizer.tiltPub.publish(tiltMsg);
                visualizer.bdrPub.publish(bdrMsg);

                visualizer.visualizeSphere(traj.getPos(delta),
                                           config.dilateRadius);
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
