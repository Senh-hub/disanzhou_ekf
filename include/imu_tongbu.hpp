#pragma once
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace week3 {
struct Imu {double t;cv::Vec4d q;};
inline std::vector<Imu> read_imu(const std::string& path){
 std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open IMU: "+path);
 std::vector<Imu> a;std::string line;
 while(std::getline(f,line)){if(line.empty())continue;std::istringstream ss(line);Imu v;
  if(!(ss>>v.t>>v.q[0]>>v.q[1]>>v.q[2]>>v.q[3]))throw std::runtime_error("Bad IMU row");
  double n=cv::norm(v.q);if(!std::isfinite(v.t)||!std::isfinite(n)||n<1e-8||(!a.empty()&&v.t<=a.back().t))throw std::runtime_error("Invalid IMU time/quaternion");
  v.q/=n;a.push_back(v);
 }if(a.size()<2)throw std::runtime_error("Need >=2 IMU samples");return a;
}
inline bool interpolate(const std::vector<Imu>& a,double t,cv::Vec4d& q){
 if(t<a.front().t||t>a.back().t)return false; // No fabricated extrapolation at boundaries.
 auto it=std::lower_bound(a.begin(),a.end(),t,[](const Imu& v,double x){return v.t<x;});
 if(it==a.begin()){q=it->q;return true;}auto b=*it,c=*(it-1);double u=(t-c.t)/(b.t-c.t),d=c.q.dot(b.q);
 if(d<0){b.q=-b.q;d=-d;}d=std::clamp(d,-1.,1.);
 if(d>.9995)q=(1-u)*c.q+u*b.q;else {double th=acos(d);q=(sin((1-u)*th)*c.q+sin(u*th)*b.q)/sin(th);}
 q/=cv::norm(q);return true;
}
inline cv::Matx33d quat_rotation(cv::Vec4d q){
 double w=q[0],x=q[1],y=q[2],z=q[3];return {1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
}
}
