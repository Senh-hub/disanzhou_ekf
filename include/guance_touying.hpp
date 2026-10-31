#pragma once
#include <opencv2/calib3d.hpp>
#include "zhengche_ekf.hpp"
#include "leixing.hpp"
namespace week3 {
inline std::vector<cv::Point3d> object_points(double w,double h){return {{-w/2,-h/2,0},{-w/2,h/2,0},{w/2,h/2,0},{w/2,-h/2,0}};}
inline cv::Mat pose_observation(const cv::Vec3d& r,const cv::Vec3d& t,const cv::Matx33d& rwc,const cv::Vec3d& twc){
 cv::Matx33d rc;cv::Rodrigues(r,rc);cv::Vec3d p=rwc*t+twc,n=-(rwc*rc*cv::Vec3d(0,0,1));
 return (cv::Mat_<double>(4,1)<<p[0],p[1],p[2],atan2(n[1],n[0]));
}
inline std::vector<Guance> observations(const Zhuangjiaban& b,int id,double width,double height,const cv::Mat& K,const cv::Mat& D,const cv::Matx33d& rwc,const cv::Vec3d& twc,double sigma_px,double max_error){
 auto obj=object_points(width,height);std::vector<cv::Point2d> img;for(auto p:b.dian)img.emplace_back(p.x,p.y);
 std::vector<cv::Mat> rs,ts;cv::solvePnPGeneric(obj,img,K,D,rs,ts,false,cv::SOLVEPNP_IPPE);
 std::vector<Guance> out;
 for(size_t k=0;k<rs.size();++k){cv::Vec3d r(rs[k].at<double>(0),rs[k].at<double>(1),rs[k].at<double>(2)),t(ts[k].at<double>(0),ts[k].at<double>(1),ts[k].at<double>(2));
  if(!cv::checkRange(rs[k])||!cv::checkRange(ts[k]))continue;
  cv::Matx33d rot;cv::Rodrigues(r,rot);bool valid=true;for(auto p:obj)if((rot*cv::Vec3d(p.x,p.y,p.z)+t)[2]<=.01)valid=false;if(!valid)continue;
  std::vector<cv::Point2d> proj;cv::Mat jac;cv::projectPoints(obj,r,t,K,D,proj,jac);double e=0;for(int j=0;j<4;++j)e+=cv::norm(proj[j]-img[j])*cv::norm(proj[j]-img[j]);e=sqrt(e/4);if(e>max_error)continue;
  cv::Vec3d n=-(rwc*rot*cv::Vec3d(0,0,1));if(hypot(n[0],n[1])<.2)continue;
  // Linearize pixel projection in six pose parameters. Use SVD pseudoinverse,
  // then propagate covariance to world xyz/yaw. Floors cover unmodelled errors.
  cv::Mat j=jac.colRange(0,6).clone(), inv;cv::invert(j.t()*j,inv,cv::DECOMP_SVD);cv::Mat cp=inv*(sigma_px*sigma_px);
  cv::Mat g=cv::Mat::zeros(4,6,CV_64F);double eps=1e-5;
  for(int c=0;c<6;++c){auto rp=r,rm=r,tp=t,tm=t;if(c<3){rp[c]+=eps;rm[c]-=eps;}else {tp[c-3]+=eps;tm[c-3]-=eps;}
   cv::Mat d=pose_observation(rp,tp,rwc,twc)-pose_observation(rm,tm,rwc,twc);d.at<double>(3)=wrap(d.at<double>(3));d/=2*eps;d.copyTo(g.col(c));}
  cv::Mat rr=g*cp*g.t();for(int c=0;c<3;++c)rr.at<double>(c,c)+=.0004;rr.at<double>(3,3)+=.0225;rr=(rr+rr.t())*.5;
  out.push_back({pose_observation(r,t,rwc,twc),rr,id,e});
 }
 std::sort(out.begin(),out.end(),[](const Guance& a,const Guance& b){return a.reproj<b.reproj;});return out;
}
inline std::vector<cv::Point2d> project_plate(const ZhengcheEkf& ekf,int j,double width,double height,double tilt,const cv::Matx33d& rwc,const cv::Vec3d& twc,const cv::Mat& K,const cv::Mat& D){
 cv::Mat z=ekf.observe(ekf.x,j);double th=z.at<double>(3);cv::Vec3d center(z.at<double>(0),z.at<double>(1),z.at<double>(2));
 cv::Vec3d right(-sin(th),cos(th),0),down(sin(tilt)*cos(th),sin(tilt)*sin(th),-cos(tilt));
 std::vector<cv::Point3d> camera;for(auto p:object_points(width,height)){cv::Vec3d c=rwc.t()*(center+p.x*right+p.y*down-twc);if(c[2]<.05)return {};camera.emplace_back(c[0],c[1],c[2]);}
 std::vector<cv::Point2d> uv;cv::projectPoints(camera,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),K,D,uv);return uv;
}
}
