#pragma once

#include<string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

#include "opencv2/dnn.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/face/facerec.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.hpp"
#include "tesseract/baseapi.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <tbb/mutex.h>

namespace hai {

  class VisualContextAnnotator {
   public:
      VisualContextAnnotator();

      virtual ~VisualContextAnnotator();

      void loadCascadeClassifier(const std::string cascadeClassifierPath);

      void loadLBPModel(const std::string path, double maxDistance = 65.0);

      void loadCAFFEModel(const std::string modelBinPath, const std::string modelProtoTextPath, const std::string synthWordPath);

      void loadTESSERACTModel(const std::string &dataPath, const std::string &lang,
                              tesseract::OcrEngineMode ocrMode = tesseract::OEM_TESSERACT_CUBE_COMBINED);

      void train(std::vector<cv::Mat> samples, int label, std::string ontology) noexcept;

      void update(std::vector<cv::Mat> samples, int label, std::string ontology) noexcept;

      std::vector<cv::Rect> detectWithCascadeClassifier(const cv::Mat &frame_gray, const cv::Size &minSize = cv::Size(80, 80)) noexcept;

      std::vector<cv::Rect> detectWithMorphologicalGradient(const cv::Mat &frame, const cv::Size &minSize = cv::Size(8, 8),
                                                   cv::Size kernelSize = cv::Size(9, 1)) noexcept;

      std::vector<cv::Rect>
      detectObjectsWithCanny(const cv::Mat &frame_gray, const double &lowThreshold = 77,
                             const cv::Size &minSize = cv::Size(80, 80),const cv::Size &maxSize = cv::Size(400, 400)) noexcept;

      std::vector<Annotation>
      detectContoursWithCanny(const cv::Mat &frame_gray, const double &lowThreshold = 77,
                              const cv::Size &minSize = cv::Size(80, 80)) noexcept;

      std::vector<Annotation> predictWithLBP(const cv::Mat &frame_gray) noexcept;

      std::vector<Annotation>
      predictWithLBP(const std::vector<cv::Rect> &detects, const cv::Mat &frame_gray, const std::string &annotationType) noexcept;

      Annotation
      predictWithLBPInRectangle(const cv::Rect &detect, const cv::Mat &frame_gray, const std::string &annotationType) noexcept;

      std::vector<Annotation> predictWithCAFFE(const cv::Mat frame, const cv::Mat frame_gray) noexcept;

      std::vector<Annotation> predictWithCAFFE(const std::vector<cv::Rect> &detects, const cv::Mat &frame) noexcept;

      Annotation predictWithCAFFEInRectangle(const cv::Rect &detect, const cv::Mat &frame)noexcept;

      std::vector<Annotation> predictWithTESSERACT(const cv::Mat &frame_gray) noexcept;

      std::vector<Annotation> predictWithTESSERACT(const std::vector<cv::Rect> &detects, const cv::Mat &frame_gray) noexcept;

      Annotation predictWithTESSERACTInRectangle(const cv::Rect &detect, const cv::Mat &frame_gray) noexcept;

      /**
     * Waits timeMillisToWait and the checks if key was pressed
     * This method uses global lock (static mutex)
     * @param timeMillisToWait milliseconds to wait before check for key
     * @param key int value
     * @return true if key was pressed
     */
      static bool checkKeyWasPressed(const int timeMillisToWait, const int key) noexcept;

      /**
       * Calls cv::imshow with global lock
       * Draws: frame in window with: name
       * @param frame cv::Mat to be shown
       */
      static void showImage(const std::string name, const cv::Mat &frame);

   private:
      cv::Ptr<cv::face::FaceRecognizer> model;
      std::unique_ptr<cv::CascadeClassifier> cascade_classifier;
      std::unique_ptr<tesseract::TessBaseAPI> tess;
      std::unique_ptr<cv::dnn::Net> net;
      tbb::mutex cascadeClassLock, lbpLock, lbpLock1, caffeLock, caffeLock1, tessLock, tessLock1,
        morphGradientLock, contoursWithCannyLock, objectsWithCannyLock, training, training2;

      double maxDistance;
      std::string lbpModelPath;

      void getMaxClass(cv::Mat& probBlob, int& classId, double& classProb);

      std::vector<std::string> readClassNames(const std::string filename);

      std::vector<std::string> classNames;

      static tbb::mutex wait_key_mutex, imshow_mutex;
  };
}
