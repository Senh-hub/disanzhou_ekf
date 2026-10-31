#include <iostream>
#include <random>
#include "imu_tongbu.hpp"
#include "guance_touying.hpp"
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){try{
 cv::setNumThreads(1);
 using namespace week3;
 std::vector<Imu> a={{0,{1,0,0,0}},{1,{-sqrt(.5),0,0,-sqrt(.5)}}};cv::Vec4d q;
 require(interpolate(a,.5,q),"interpolate");auto r=quat_rotation(q);auto v=r*cv::Vec3d(1,0,0);require(fabs(v[0]-sqrt(.5))<1e-9,"SLERP/sign");require(!interpolate(a,-.009,q),"boundary no extrapolation");
 ZhengcheEkf ek;ek.x=(cv::Mat_<double>(11,1)<<2,.1,1,.2,.5,0,3.13,1,.25,.3,.06);
 for(int j=0;j<4;++j){cv::Mat analytic=ek.H(j);for(int i=0;i<11;++i){cv::Mat xp=ek.x.clone(),xm=ek.x.clone();xp.at<double>(i)+=1e-6;xm.at<double>(i)-=1e-6;cv::Mat d=(ek.observe(xp,j)-ek.observe(xm,j))/(2e-6);require(cv::norm(d-analytic.col(i))<1e-6,"observation Jacobian");}}
 ZhengcheEkf truth,track;truth.x=(cv::Mat_<double>(11,1)<<3,.12,1,-.04,.5,0,2.8,1.4,.25,.3,.06);truth.ready=true;truth.P=cv::Mat::eye(11,11,CV_64F)*1e-8;
 std::mt19937 gen(42);std::normal_distribution<double> noise(0,1);int accepted=0;
 for(int k=0;k<600;++k){truth.predict(.04);track.predict(.04);std::vector<Guance> gs;
  if(k%100<90){for(int l=0;l<2;++l){int j=(k/18+l)%4;cv::Mat z=truth.observe(truth.x,j),rr=cv::Mat::eye(4,4,CV_64F)*.0001;rr.at<double>(3,3)=.0025;for(int c=0;c<4;++c)z.at<double>(c)+=sqrt(rr.at<double>(c,c))*noise(gen);z.at<double>(3)=wrap(z.at<double>(3));gs.push_back({z,rr,l,0});}}
  accepted+=track.update(gs);require(cv::checkRange(track.x),"finite state");if(track.ready){require(cv::norm(track.P-track.P.t())<1e-8,"symmetric covariance");cv::Mat eig;cv::eigen(track.P,eig);require(eig.at<double>(10)>-1e-9,"PSD covariance");}
 }
 std::cout << "accepted=" << accepted << " estimate=" << track.x.t() << " truth=" << truth.x.t() << "\n";
 require(accepted>900,"switch association acceptance");require(hypot(track.s(0)-truth.s(0),track.s(2)-truth.s(2))<.08,"center convergence");require(fabs(track.s(7)-truth.s(7))<.2,"angular velocity convergence");
 // Projection -> PnP -> world observation round trip.
 cv::Mat K=(cv::Mat_<double>(3,3)<<1000,0,640,0,1000,480,0,0,1),D=cv::Mat::zeros(1,5,CV_64F);
 cv::Matx33d rwc(0,0,1,-1,0,0,0,-1,0);ZhengcheEkf model;model.x=(cv::Mat_<double>(11,1)<<3,0,0,0,0,0,3.0,0,.25,.25,0);
 auto uv=project_plate(model,0,.135,.055,0,rwc,{0,0,0},K,D);Zhuangjiaban b;for(int i=0;i<4;++i)b.dian[i]=uv[i];
 auto obs=observations(b,0,.135,.055,K,D,rwc,{0,0,0},1.5,5);require(!obs.empty(),"PnP observation");require(cv::norm(obs[0].z.rowRange(0,3)-model.observe(model.x,0).rowRange(0,3))<.002,"roundtrip position");
 std::cout<<"PASS: SLERP, boundaries, Jacobians, 600-frame switches/misses, PSD, PnP roundtrip. Accepted="<<accepted<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
