#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace week3 {
inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }
struct Guance { cv::Mat z, R; int detection=0; double reproj=0; };
// State: cx,vx,cy,vy,z0,vz,yaw,omega,r0,r1,dz. Opposite plates share radius/height.
class ZhengcheEkf {
public:
 cv::Mat x=cv::Mat::zeros(11,1,CV_64F), P; bool ready=false;
 double q_acc=.2, q_alpha=2., q_radius=1e-5, q_height=1e-5, gate=18.47;
 double last_nis=std::numeric_limits<double>::quiet_NaN(); int matched=-1, misses=0;
 double& s(int i) { return x.at<double>(i); }
 double s(int i) const { return x.at<double>(i); }
 cv::Mat observe(const cv::Mat& a,int j) const {
  double th=a.at<double>(6)+j*CV_PI/2, r=a.at<double>(8+j%2);
  return (cv::Mat_<double>(4,1)<<a.at<double>(0)+r*cos(th),a.at<double>(2)+r*sin(th),a.at<double>(4)+(j%2?a.at<double>(10):0),th);
 }
 cv::Mat H(int j) const {
  cv::Mat h=cv::Mat::zeros(4,11,CV_64F); double th=s(6)+j*CV_PI/2,r=s(8+j%2);
  h.at<double>(0,0)=h.at<double>(1,2)=h.at<double>(2,4)=h.at<double>(3,6)=1;
  h.at<double>(0,6)=-r*sin(th); h.at<double>(1,6)=r*cos(th);
  h.at<double>(0,8+j%2)=cos(th);h.at<double>(1,8+j%2)=sin(th);
  if(j%2) h.at<double>(2,10)=1; return h;
 }
 void init(const Guance& g) {
  x=cv::Mat::zeros(11,1,CV_64F); s(8)=s(9)=.25; s(6)=g.z.at<double>(3);
  s(0)=g.z.at<double>(0)-.25*cos(s(6));s(2)=g.z.at<double>(1)-.25*sin(s(6));s(4)=g.z.at<double>(2);
  P=cv::Mat::zeros(11,11,CV_64F);
  double d[]={.25,4,.25,4,.09,1,.25,9,.04,.04,.04};
  for(int i=0;i<11;++i)P.at<double>(i,i)=d[i];ready=true;misses=0;
 }
 void predict(double dt) {
  if(!ready)return;
  if(!(dt>0 && dt<1)) {ready=false;return;}
  cv::Mat f=cv::Mat::eye(11,11,CV_64F),q=cv::Mat::zeros(11,11,CV_64F);
  for(int k: {0,2,4,6}) {
   f.at<double>(k,k+1)=dt;double v=k==6?q_alpha:q_acc;
   // Continuous white acceleration spectral density; exact integration.
   q.at<double>(k,k)=v*dt*dt*dt/3;q.at<double>(k,k+1)=q.at<double>(k+1,k)=v*dt*dt/2;q.at<double>(k+1,k+1)=v*dt;
  }
  q.at<double>(8,8)=q.at<double>(9,9)=q_radius*dt;q.at<double>(10,10)=q_height*dt;
  x=f*x;P=f*P*f.t()+q;matched=-1;last_nis=std::numeric_limits<double>::quiet_NaN();
 }
 int update(const std::vector<Guance>& gs) {
  if(gs.empty()){++misses;return 0;}
  if(!ready){init(gs.front());matched=0;return 1;}
  // Greedy minimum Mahalanobis association, one measurement per detection and plate.
  std::vector<int> used;bool plates[4]={false,false,false,false};int count=0;
  for(int pass=0;pass<4;++pass){double best=gate;int bi=-1,bj=-1;cv::Mat by,bh,bs;
   for(int i=0;i<(int)gs.size();++i){if(std::find(used.begin(),used.end(),gs[i].detection)!=used.end())continue;
    for(int j=0;j<4;++j){if(plates[j])continue;cv::Mat y=gs[i].z-observe(x,j);y.at<double>(3)=wrap(y.at<double>(3));
     cv::Mat h=H(j),ss=h*P*h.t()+gs[i].R,sol; if(!cv::solve(ss,y,sol,cv::DECOMP_CHOLESKY))continue;
     double nis=y.dot(sol);if(nis<best){best=nis;bi=i;bj=j;by=y;bh=h;bs=ss;}
    }
   }
   if(bi<0)break;cv::Mat kt;cv::solve(bs,bh*P,kt,cv::DECOMP_CHOLESKY);cv::Mat k=kt.t();
   cv::Mat nx=x+k*by;
   if(nx.at<double>(8)<.05||nx.at<double>(8)>.8||nx.at<double>(9)<.05||nx.at<double>(9)>.8||fabs(nx.at<double>(10))>.5){used.push_back(gs[bi].detection);continue;}
   x=nx;cv::Mat a=cv::Mat::eye(11,11,CV_64F)-k*bh;
   P=a*P*a.t()+k*gs[bi].R*k.t();P=(P+P.t())*.5;
   used.push_back(gs[bi].detection);plates[bj]=true;last_nis=best;matched=bj;++count;
  }
  misses=count?0:misses+1;return count;
 }
};
}
