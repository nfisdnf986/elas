#include <iomanip>
#include <unistd.h>
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include <cv.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <HAL/Messages/Image.h>
#include <HAL/Messages/Logger.h>
#include <HAL/Messages/Reader.h>

#include "elas.h"
#include "image.h"

using namespace std;


/**
 * logtool functionality
 *
 * - [X] Extract a given set of frames
 * - [X] Extract a log into individual images
 * - [X] Extract imu from a log and save it as csv
 * - [X] Extract posys from a log and save it as csv
 * - [X] Concatenate multiple logs
 * - [ ] Concatenate multiple single images into a log
 * - [ ] Reorder a log by timestamp
 * - [ ] Add an index
 * - [ ] Remove an index
 * - [ ] Output index, header to human-readable format
 */

DEFINE_string(in, "", "Input log file or input directory.");
DEFINE_string(out, "", "Output log file or output directory.");

DEFINE_bool(extract_log, false, "Enable log subset extraction.");
DEFINE_bool(extract_images, false, "Enable image extraction to individual files.");
DEFINE_bool(extract_imu, false, "Enable IMU extraction to individual files.");
DEFINE_bool(extract_posys, false, "Enable Posys extraction to individual files.");
DEFINE_string(extract_types, "",
              "Comma-separated list of types to extract from log. "
              "Options include \"cam\", \"imu\", and \"posys\".");
DEFINE_string(extract_frame_range, "",
              "Range (inclusive) of image frames to extract from the log. "
              "Should be a comma-separated pair, e.g \"0,200\"");

DEFINE_string(cat_logs, "",
              "Comma-separated list of logs to concatenate together. ");



typedef std::function<bool(char)> TrimPred;

inline std::string& LTrim(std::string& s, const TrimPred& pred) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), pred));
  return s;
}

inline std::string& RTrim(std::string& s, const TrimPred& pred) {
  s.erase(std::find_if(s.rbegin(), s.rend(), pred).base(), s.end());
  return s;
}

inline std::string& TrimQuotes(std::string& s) {
  TrimPred qpred = [](char c) { return c != '\'' && c != '\"'; };
  return LTrim(RTrim(s, qpred), qpred);
}


/**
 * Given a delim-separated string, split it into component elements.
 *
 * Use a stringstream to convert to the destination type.
 */
template <typename T>
inline void Split(const std::string& s, char delim,
                  std::vector<T>* elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    T val;
    std::stringstream val_ss(item);
    val_ss >> val;
    elems->push_back(val);
  }
}

/**
 * Special case for string splitting, since it doesn't need an extra
 * re-parsing through a stringstream.
 */
template<>
inline void Split<std::string>(const std::string& s, char delim,
                               std::vector<std::string>* elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems->push_back(item);
  }
}

/** compute disparities of pgm image input pair file_1, file_2 */
void process (const char* file_1,const char* file_2) {

  cout << "Processing: " << file_1 << ", " << file_2 << endl;

  // load images
  image<uchar> *I1,*I2;
  I1 = loadPGM(file_1);
  I2 = loadPGM(file_2);

  // check for correct size
  if (I1->width()<=0 || I1->height() <=0 || I2->width()<=0 || I2->height() <=0 ||
      I1->width()!=I2->width() || I1->height()!=I2->height()) {
    cout << "ERROR: Images must be of same size, but" << endl;
    cout << "       I1: " << I1->width() <<  " x " << I1->height() << 
                 ", I2: " << I2->width() <<  " x " << I2->height() << endl;
    delete I1;
    delete I2;
    return;    
  }

  // get image width and height
  int32_t width  = I1->width();
  int32_t height = I1->height();

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  float* D1_data = (float*)malloc(width*height*sizeof(float));
  float* D2_data = (float*)malloc(width*height*sizeof(float));

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  Elas elas(param);
  elas.process(I1->data,I2->data,D1_data,D2_data,dims);

  // find maximum disparity for scaling output disparity images to [0..255]
  float disp_max = 0;
  for (int32_t i=0; i<width*height; i++) {
    if (D1_data[i]>disp_max) disp_max = D1_data[i];
    if (D2_data[i]>disp_max) disp_max = D2_data[i];
  }

  // copy float to uchar
  image<uchar> *D1 = new image<uchar>(width,height);
  image<uchar> *D2 = new image<uchar>(width,height);
  for (int32_t i=0; i<width*height; i++) {
    D1->data[i] = (uint8_t)max(255.0*D1_data[i]/disp_max,0.0);
    D2->data[i] = (uint8_t)max(255.0*D2_data[i]/disp_max,0.0);
  }

  // save disparity images
  char output_1[1024];
  char output_2[1024];
  strncpy(output_1,file_1,strlen(file_1)-4);
  strncpy(output_2,file_2,strlen(file_2)-4);
  output_1[strlen(file_1)-4] = '\0';
  output_2[strlen(file_2)-4] = '\0';
  strcat(output_1,"_disp.pgm");
  strcat(output_2,"_disp.pgm");
  savePGM(D1,output_1);
  savePGM(D2,output_2);

  // free memory
  delete I1;
  delete I2;
  delete D1;
  delete D2;
  free(D1_data);
  free(D2_data);
}


/** Save individual file based on pb:Image type. */
inline void SaveImage(const std::string& out_dir,
                      int channel_index,
                      unsigned int frame_number,
                      double timestamp,
                      const hal::ImageMsg& image) {

  // Convert index to string.
  std::string index;
  std::ostringstream convert;
  convert << channel_index;
  index = convert.str();

  std::string file_prefix = out_dir + "/";
  file_prefix = file_prefix + "channel" + index;

  convert.str("");
  convert.clear();
  if (timestamp == 0.)
    convert << std::fixed << std::setfill('0') << std::setw(5) << frame_number;
  else
    convert << std::fixed << std::setfill('0') << std::setw(5) << frame_number
            << "_" << std::setprecision(9) << timestamp;
  index = convert.str();

  std::string filename;

  // Use OpenCV to handle saving the file for us.
  cv::Mat cv_image = hal::WriteCvMat(image);

  if (image.type() == hal::Type::PB_FLOAT) {
    // Save floats to our own "portable depth map" format.
    filename = file_prefix + "_" + index + ".pdm";

    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    file << "P7" << std::endl;
    file << cv_image.cols << " " << cv_image.rows << std::endl;
    const size_t size = cv_image.elemSize1() * cv_image.rows * cv_image.cols;
    file << 4294967295 << std::endl;
    file.write((const char*)cv_image.data, size);
    file.close();
  } else if (image.type() == hal::Type::PB_BYTE
             || image.type() == hal::Type::PB_UNSIGNED_BYTE
             || image.type() == hal::Type::PB_SHORT
             || image.type() == hal::Type::PB_UNSIGNED_SHORT) {
    // OpenCV only supports byte/short data types with 1/3 channel images.
    filename = file_prefix + "_" + index + ".pgm";
    cv::imwrite(filename, cv_image);

    if (file_prefix.back() == '1')
    {
        int whwqehqw = file_prefix.size();
        std::string cam0 = filename;
        cam0[whwqehqw-1] = '0';

        process(cam0.c_str(), filename.c_str());
    }
  } else {
    LOG(FATAL) << "Input image type not supported for extraction.";
  }
}


/** Extracts single images out of a log file. */
void ExtractImages() {
  static const int kNoRange = -1;

  int frame_min = kNoRange, frame_max = kNoRange;
  std::vector<int> frames;
  Split(TrimQuotes(FLAGS_extract_frame_range), ',', &frames);
  if (!frames.empty()) {
    CHECK_EQ(2, frames.size()) << "extract_frame_range must be frame PAIR";
    frame_min = frames[0];
    frame_max = frames[1];
    CHECK_LE(frame_min, frame_max)
        << "Minimum frame index must be <= than max frame index.";
  }

  hal::Reader reader(FLAGS_in);
  reader.Enable(hal::Msg_Type_Camera);

  int idx = 0;
  std::unique_ptr<hal::Msg> msg;
  while (frame_min != kNoRange && idx < frame_min) {
    if ((msg = reader.ReadMessage()) && msg->has_camera()) {
      ++idx;
    }
  }

  while ((frame_max == kNoRange ||
          idx <= frame_max) &&
         (msg = reader.ReadMessage())) {
    if (msg->has_camera()) {
      const hal::CameraMsg& cam_msg = msg->camera();
      for (int ii = 0; ii < cam_msg.image_size(); ++ii) {
        const hal::ImageMsg& img_msg = cam_msg.image(ii);
        SaveImage(FLAGS_out, ii, idx, cam_msg.system_time(), img_msg);
      }
      ++idx;
    }
  }
}

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_extract_images != 1) {
    LOG(FATAL) << "Must choose extract_images logtool task.";
  }

  if (FLAGS_extract_images) {
    CHECK(!FLAGS_in.empty()) << "Input file required for extraction.";
    CHECK(!FLAGS_out.empty()) << "Output directory required for extraction.";
    ExtractImages();
  }

  return 0;
}
