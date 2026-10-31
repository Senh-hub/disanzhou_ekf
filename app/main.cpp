#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <opencv2/opencv.hpp>
#include "nn_shibie.hpp"
#include "xiangji_canshu.hpp"
#include "imu_tongbu.hpp"
#include "guance_touying.hpp"

int main(int argc,char** argv){try{
 cv::setNumThreads(1);
 std::string cfg=argc>1?argv[1]:"config/disanzhou.yaml";cv::FileStorage fs(cfg,cv::FileStorage::READ);if(!fs.isOpened())throw std::runtime_error("Cannot open "+cfg);
 auto str=[&](const char* key){std::string s;fs[key]>>s;return s;};
 auto num=[&](const char* key,double def){return fs[key].empty()?def:(double)fs[key];};
 bool assumed=num("assumptions_confirmed",0)==0;
 const std::string mode=str("time_mode");auto imu=week3::read_imu(str("imu"));std::vector<double> times;
 if(mode=="frame_rows_assumed"){for(auto v:imu)times.push_back(v.t);}
 else if(mode=="timestamps_file"){std::ifstream f(str("image_timestamps"));double t;while(f>>t)times.push_back(t);if(times.empty())throw std::runtime_error("No image timestamps");}
 else throw std::runtime_error("Unknown time_mode");
 for(size_t i=0;i<times.size();++i)if(!std::isfinite(times[i])||(i&&times[i]<=times[i-1]))throw std::runtime_error("Invalid image timestamps");
 cv::Mat ext;fs["R_imu_camera"]>>ext;cv::Matx33d ric;if(ext.rows!=3||ext.cols!=3)throw std::runtime_error("Need R_imu_camera 3x3");ext.convertTo(ext,CV_64F);for(int i=0;i<9;++i)ric.val[i]=ext.at<double>(i/3,i%3);
 if(cv::norm(cv::Mat(ric.t()*ric-cv::Matx33d::eye()))>1e-5||fabs(cv::determinant(ric)-1)>1e-5)throw std::runtime_error("Invalid extrinsic rotation");
 cv::Mat trans;fs["t_imu_camera"]>>trans;cv::Vec3d tic(0,0,0);if(trans.total()==3){trans.convertTo(trans,CV_64F);for(int i=0;i<3;++i)tic[i]=trans.at<double>(i);}
 auto c=du_xiangji_canshu(str("camera"));cv::VideoCapture cap(str("video"));if(!cap.isOpened())throw std::runtime_error("Cannot open video");
 int w=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH),h=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
 if(w!=c.tupian_kuan||h!=c.tupian_gao)throw std::runtime_error("Image resolution differs from calibrated configuration");
 NnShibie detector(str("model"),.5f,.2f);if(!detector.keyong())throw std::runtime_error("Model unavailable");
 std::string dest=str("output");std::filesystem::create_directories(dest);
 double outfps=(times.size()-1)/(times.back()-times.front());
 cv::VideoWriter writer(dest+"/zhengche_result.mp4",cv::VideoWriter::fourcc('m','p','4','v'),outfps,cv::Size(w,h));if(!writer.isOpened())throw std::runtime_error("Cannot create video");
 std::ofstream csv(dest+"/zhuangtai.csv");if(!csv)throw std::runtime_error("Cannot create CSV");
 csv<<"frame,time,sync,valid,segment,matched,updates,nis,cx,vx,cy,vy,z0,vz,yaw,omega,r0,r1,dz";for(int i=0;i<11;++i)csv<<",sd"<<i;csv<<"\n"<<std::setprecision(12);
 week3::ZhengcheEkf ekf;ekf.q_acc=num("q_acc",.2);ekf.q_alpha=num("q_alpha",2);ekf.gate=num("nis_gate",18.47);
 double offset=num("offset_s",.009),last_hit=-1;int frame_id=0,segment=0,total_updates=0,sync_skips=0;
 std::string target=str("target_number");cv::Mat frame;
 while(cap.read(frame)){
  if(frame_id>=(int)times.size())throw std::runtime_error("More video frames than timestamps");double t=times[frame_id];cv::Vec4d q;bool sync=week3::interpolate(imu,t-offset,q);int updates=0;
  if(frame_id)ekf.predict(t-times[frame_id-1]);
  if(ekf.ready&&last_hit>=0&&t-last_hit>num("lost_timeout_s",.6))ekf.ready=false;
  cv::Matx33d rwc=cv::Matx33d::eye();cv::Vec3d twc(0,0,0);
  if(sync){auto rwi=week3::quat_rotation(q);if(num("quaternion_inverse",0))rwi=rwi.t();rwc=rwi*ric;twc=rwi*tic;
   auto bans=detector.jiance(frame);std::vector<week3::Guance> gs;
   // Lock to one identity. Four-plate model excludes outpost/base/sentry.
   if(target.empty()&&!ekf.ready){float best=0;for(auto& b:bans)if(b.color==(int)num("target_color",1)&&(b.number=="3"||b.number=="4"||b.number=="5"||b.number=="hero"||b.number=="engineer")&&b.score>best){target=b.number;best=b.score;}}
   int id=0;for(auto& b:bans){if(b.color!=(int)num("target_color",1)||b.number!=target)continue;
    double width=target=="hero"?num("large_width_m",.225):num("small_width_m",.135);
    auto obs=week3::observations(b,id++,width,num("height_m",.055),c.neican_juzhen,c.jibian_canshu,rwc,twc,num("pixel_sigma",1.5),num("max_reproj_px",5));gs.insert(gs.end(),obs.begin(),obs.end());
    for(int k=0;k<4;++k)cv::line(frame,b.dian[k],b.dian[(k+1)%4],{0,255,0},2);
   }
   bool before=ekf.ready;updates=ekf.update(gs);if(!before&&ekf.ready)++segment;if(updates){last_hit=t;total_updates+=updates;}
   if(ekf.ready){double width=target=="hero"?num("large_width_m",.225):num("small_width_m",.135);
    for(int j=0;j<4;++j){auto uv=week3::project_plate(ekf,j,width,num("height_m",.055),num("plate_tilt_deg",0)*CV_PI/180,rwc,twc,c.neican_juzhen,c.jibian_canshu);
     if(uv.size()!=4)continue;bool ok=true;for(auto p:uv)if(!std::isfinite(p.x)||!std::isfinite(p.y)||fabs(p.x)>1e6||fabs(p.y)>1e6)ok=false;if(!ok)continue;
     cv::Scalar color=j==ekf.matched?cv::Scalar(0,210,255):cv::Scalar(255,160,30);
     for(int k=0;k<4;++k){cv::Point a=uv[k],b=uv[(k+1)%4];if(cv::clipLine(frame.size(),a,b))cv::line(frame,a,b,color,2);}
     if(uv[0].x>=0&&uv[0].x<w&&uv[0].y>=0&&uv[0].y<h)cv::putText(frame,"P"+std::to_string(j),uv[0],cv::FONT_HERSHEY_SIMPLEX,.6,color,2);
    }
   }
  }else ++sync_skips;
  csv<<frame_id<<','<<t<<','<<sync<<','<<ekf.ready<<','<<segment<<','<<ekf.matched<<','<<updates<<','<<ekf.last_nis;
  for(int i=0;i<11;++i)csv<<','<<(ekf.ready?ekf.s(i):std::numeric_limits<double>::quiet_NaN());for(int i=0;i<11;++i)csv<<','<<(ekf.ready?sqrt(std::max(0.,ekf.P.at<double>(i,i))):std::numeric_limits<double>::quiet_NaN());csv<<'\n';
  int left=std::max(0,w-455);cv::rectangle(frame,{left,0},{w,std::min(h,390)},{20,20,20},cv::FILLED);
  auto line=[&](int i,const std::string& s,cv::Scalar color=cv::Scalar(240,240,240)){cv::putText(frame,s,{left+12,25+i*26},cv::FONT_HERSHEY_SIMPLEX,.58,color,1,cv::LINE_AA);};
  line(0,cv::format("t=%.3f frame=%d target=%s",t,frame_id,target.c_str()));line(1,assumed?"ASSUMED TIME / EXTRINSICS":"CONFIG CONFIRMED",{0,180,255});line(2,!sync?"NO IMU BRACKET":(!ekf.ready?"LOST / INITIALIZING":(updates?"TRACKING":"PREDICT ONLY")));
  const char* names[]={"cx (m)","vx (m/s)","cy (m)","vy (m/s)","z0 (m)","vz (m/s)","yaw (rad)","omega (rad/s)","r0 (m)","r1 (m)","dz (m)"};
  for(int i=0;i<11;++i)line(i+3,ekf.ready?cv::format("%s = %.4f",names[i],ekf.s(i)):std::string(names[i])+" = --");
  writer.write(frame);++frame_id;if(frame_id%50==0)std::cout<<"frame "<<frame_id<<" segment "<<segment<<" updates "<<total_updates<<std::endl;
 }
 if(frame_id!=(int)times.size())throw std::runtime_error("Video/timestamp count mismatch");
 std::ofstream summary(dest+"/run_summary.txt");summary<<"frames="<<frame_id<<"\nupdates="<<total_updates<<"\nsegments="<<segment<<"\nsync_skips="<<sync_skips<<"\nassumptions="<<assumed<<"\ntarget="<<target<<"\ntime_mode="<<mode<<"\n";
 std::cout<<"Done: "<<dest<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
