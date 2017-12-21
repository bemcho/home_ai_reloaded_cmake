#pragma once

#include <functional>
#include <random>
#include <string>
#include <thread>
#include "tbb/parallel_invoke.h"
#include "tbb/mutex.h"

#include "ClipsAdapter.hpp"
#include "VisualContextAnnotator.hpp"

namespace hai {
  class VisualREPL {
   public:
      VisualREPL(string aReplName,
                 ClipsAdapter &aClips,
                 function<vector<Annotation>(cv::Mat, cv::Mat)> aAnnotationFN,
                 function<void(std::vector<cv::Mat>, int, string, bool &)> aTrainFN,
                 bool aShowWindow)
        : name{aReplName}, annotationFN{aAnnotationFN}, trainFN{aTrainFN}, clips{aClips},
          showWindow{aShowWindow} {}

      ~VisualREPL() { capture.release(); }

      VisualREPL(const VisualREPL &v)
        : name{v.name}, annotationFN{v.annotationFN}, trainFN{v.trainFN}, clips{v.clips},
          showWindow{v.showWindow} {
      }

      bool startAt(int index, int framesPerSecond) noexcept {
          capture.open(index);
          if (!capture.isOpened()) {
              capture.release();
              return false;
          }

          tVisualLoop = std::thread(&VisualREPL::startVisualLoop, this, framesPerSecond);
          tVisualLoop.detach();
          return true;
      }

      void startAt(const string &streamOrWebCamUrl, int framesPerSecond) noexcept {
          capture.open(streamOrWebCamUrl);
          if (!capture.isOpened()) {
              capture.release();
              cout << "--(!)Error opening video capture at: {" << streamOrWebCamUrl
                   << "}\n Make sure it's a live stream!" << endl;
              return;
          }

          tVisualLoop = std::thread(&VisualREPL::startVisualLoop, this, framesPerSecond);
          tVisualLoop.detach();
      }

      void startTraining(cv::Rect rectangle, int aLabel, string aOntology) {
          samples.clear();
          trainingRect = rectangle;
          trainingLabel = aLabel;
          trainingOntology = aOntology;
          localTrainFN =
            [&]
              (cv::Mat f_g) {
              tbb::mutex::scoped_lock lck{training};
              if (gatherSamplesFlag && samples.size() < samplesCount) {
                  samples.push_back(f_g(trainingRect));
                  return;
              }
              gatherSamplesFlag = false;
              startTrainingFlag = true;
              if (startTrainingFlag && !trainingInProgressFlag) {
                  startTrainingFlag = false;
                  trainFN(samples, trainingLabel, trainingOntology, trainingInProgressFlag);
              }
            };

          startTrainingFlag = true;
          gatherSamplesFlag = true;
      }

   private:
      tbb::mutex training;
      std::thread tVisualLoop, tTraining;
      VideoCapture capture;
      string name;

      function<vector<Annotation>(cv::Mat, cv::Mat)> annotationFN;
      function<void(std::vector<cv::Mat>, int, string, bool &)> trainFN;
      function<void(cv::Mat)> localTrainFN;

      ClipsAdapter &clips;

      const std::size_t samplesCount = 50;
      vector<cv::Mat> samples;
      cv::Rect trainingRect;
      string trainingOntology;
      int trainingLabel;

      bool showWindow;
      bool trainingInProgressFlag = false;
      bool gatherSamplesFlag = false;
      bool startTrainingFlag = false;
      bool beginTrainingRect = false;

      static void onMouseCallback(int event, int x, int y, int flags, void *userdata) {
          reinterpret_cast<VisualREPL *>(userdata)->onMouse(event, x, y, flags);
      }

      void onMouse(int event, int x, int y, int flags) {
          if (event==EVENT_LBUTTONDOWN && flags==EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON) {
              trainingRect = cv::Rect(x, y, 0, 0);
              beginTrainingRect = true;
              return;
          } else if (event==EVENT_MOUSEMOVE && flags==EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON) {
              trainingRect.width = abs(trainingRect.x - x);
              trainingRect.height = abs(trainingRect.y - y);
              return;
          } else if (event==EVENT_LBUTTONUP && flags==EVENT_FLAG_SHIFTKEY) {
              // Seed with a real random value, if available
              beginTrainingRect = false;
              std::random_device r;
              std::default_random_engine e1(r());
              std::uniform_int_distribution<int> uniform_dist(10, 100);
              int label = uniform_dist(e1);
              startTraining(trainingRect, label, "Object " + std::to_string(label));

          }
      }

      [[noreturn]] void startVisualLoop(int framesPerSecond) noexcept {

          capture.set(CAP_PROP_FRAME_WIDTH, 10000);
          capture.set(CAP_PROP_FRAME_HEIGHT, 10000);
          capture.set(CAP_PROP_FPS, framesPerSecond);
          capture.set(CAP_PROP_FRAME_WIDTH,
                      (capture.get(CAP_PROP_FRAME_WIDTH)/2) <= 1280 ? 1280 : capture.get(CAP_PROP_FRAME_WIDTH)/2);
          capture.set(CAP_PROP_FRAME_HEIGHT,
                      (capture.get(CAP_PROP_FRAME_HEIGHT)/2) <= 720 ? 720 : capture.get(CAP_PROP_FRAME_HEIGHT)/2);

          Mat frame, frame_gray;
          vector<Annotation> annotations;
          capture >> frame;
          if (showWindow) {
              namedWindow(name, WINDOW_AUTOSIZE);
              imshow(name, frame);
              setMouseCallback(name, VisualREPL::onMouseCallback, this);
          }
          while (true) {
              {

                  annotations.clear();
                  capture >> frame;

                  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
                  equalizeHist(frame_gray, frame_gray);
                  if (frame.empty()) {
                      cout << " --(!) No captured frame -- Break!" << endl;
                      break;
                  }
                  if ((gatherSamplesFlag || startTrainingFlag) && !trainingInProgressFlag) {
                      localTrainFN(frame_gray);
                  }

                  if (beginTrainingRect || gatherSamplesFlag || trainingInProgressFlag || startTrainingFlag) {
                      putText(frame, "Training ...", Point(trainingRect.x, trainingRect.y - 20), CV_FONT_NORMAL, 1.0,
                              CV_RGB(255, 0, 0), 1);
                      rectangle(frame, trainingRect, CV_RGB(255, 0, 0), 2);

                  } else {
                      trainingRect.width = 0;
                      trainingRect.height = 0;
                  }

                  if (!trainingInProgressFlag) {
                      annotations = annotationFN(frame, frame_gray);
                      clips.callFactCreateFN(annotations, name);
                  }
              }

              if (showWindow && !trainingInProgressFlag) {
                  vector<vector<Point>> contours;
                  for (auto &annot : annotations) {
                      if (annot.getType().compare("contour")==0) {
                          contours.push_back(annot.getContour());
                      } else {
                          putText(frame, annot.getDescription(),
                                  Point(annot.getRectangle().x, annot.getRectangle().y - 20), CV_FONT_NORMAL, 1.0,
                                  CV_RGB(0, 255, 0), 1);
                          rectangle(frame, annot.getRectangle(), CV_RGB(0, 255, 0), 1);
                      }
                  }
                  if (contours.size() > 0) {
                      drawContours(frame, contours, -1, CV_RGB(255, 213, 21), 2);
                  }

                  VisualContextAnnotator::showImage(name, frame);
              }

              if (VisualContextAnnotator::checkKeyWasPressed(1, 27)) {
                  break;
              }
          }
          capture.release();
          cv::destroyWindow(name);
          std::terminate();
      }
  };
}
