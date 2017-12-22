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
  using namespace std;
  using namespace cv;

  class VisualContextAnnotator {
   public:
      VisualContextAnnotator();

      virtual ~VisualContextAnnotator();

      void loadCascadeClassifier(const string cascadeClassifierPath);

      void loadLBPModel(const string path, double maxDistance = 65.0);

      void loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath);

      void loadTESSERACTModel(const string &dataPath, const string &lang,
                              tesseract::OcrEngineMode ocrMode = tesseract::OEM_TESSERACT_CUBE_COMBINED);

      void train(vector<cv::Mat> samples, int label, string ontology) noexcept;

      void update(vector<cv::Mat> samples, int label, string ontology) noexcept;

      vector<Rect> detectWithCascadeClassifier(const Mat &frame_gray, const Size &minSize = Size(80, 80)) noexcept;

      vector<Rect> detectWithMorphologicalGradient(const Mat &frame, const Size &minSize = Size(8, 8),
                                                   Size kernelSize = Size(9, 1)) noexcept;

      vector<Rect>
      detectObjectsWithCanny(const Mat &frame_gray, const double &lowThreshold = 77,
                             const Size &minSize = Size(80, 80)) noexcept;

      vector<Annotation>
      detectContoursWithCanny(const Mat &frame_gray, const double &lowThreshold = 77,
                              const Size &minSize = Size(80, 80)) noexcept;

      vector<Annotation> predictWithLBP(const Mat &frame_gray) noexcept;

      vector<Annotation>
      predictWithLBP(const vector<Rect> &detects, const Mat &frame_gray, const string &annotationType) noexcept;

      Annotation
      predictWithLBPInRectangle(const Rect &detect, const Mat &frame_gray, const string &annotationType) noexcept;

      vector<Annotation> predictWithCAFFE(const Mat frame, const Mat frame_gray) noexcept;

      vector<Annotation> predictWithCAFFE(const vector<Rect> &detects, const Mat &frame) noexcept;

      Annotation predictWithCAFFEInRectangle(const Rect &detect, const Mat &frame)noexcept;

      vector<Annotation> predictWithTESSERACT(const cv::Mat &frame_gray) noexcept;

      vector<Annotation> predictWithTESSERACT(const vector<Rect> &detects, const cv::Mat &frame_gray) noexcept;

      Annotation predictWithTESSERACTInRectangle(const Rect &detect, const Mat &frame_gray) noexcept;

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
      static void showImage(const string name, const Mat &frame);

   private:
      Ptr<face::FaceRecognizer> model;
      unique_ptr<CascadeClassifier> cascade_classifier;
      unique_ptr<tesseract::TessBaseAPI> tess;
      unique_ptr<dnn::Net> net;
      tbb::mutex cascadeClassLock, lbpLock, lbpLock1, caffeLock, caffeLock1, tessLock, tessLock1,
        morphGradientLock, contoursWithCannyLock, objectsWithCannyLock, training, training2;

      double maxDistance;
      string lbpModelPath;

      void getMaxClass(cv::Mat& probBlob, int& classId, double& classProb);

      std::vector<String> readClassNames(const string filename);

      std::vector<String> classNames;

      static tbb::mutex wait_key_mutex, imshow_mutex;
  };
}
