#include "tbb/parallel_invoke.h"
#include "../headers/Annotation.hpp"
#include "../headers/VisualContextAnnotator.hpp"
#include "../headers/ClipsAdapter.hpp"
#include "../headers/VisualREPL.hpp"

namespace homeaiapp {

  using namespace hai;

  std::string face_cascade_name{"cascade_frontalface.xml"};
  std::string window_name{"Home AI"};
  std::string lbp_recognizer_name{"lbphFaceRecognizer.xml"};
  std::string clips_vca_rules{"visualcontextrules.clp"};

  ClipsAdapter clips(clips_vca_rules);

  vector<shared_ptr<VisualREPL>> cameras;
  VisualContextAnnotator faceAnnotator;
  VisualContextAnnotator textAnnotator;
  VisualContextAnnotator objectsAnnotator;
  VisualContextAnnotator contoursAnnotator;
  VisualContextAnnotator caffeAnnotator;
  std::size_t MAX_CAMERAS = 5;
  bool WINDOW_SHOW = true;

  vector<Annotation> annotateFacesFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> face;

      face = faceAnnotator.predictWithLBP(f_g);

      result.insert(result.end(), face.begin(), face.end());

      return result;
  }

  vector<Annotation> annotateObjectsWithLBPFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> objects;

      objects = objectsAnnotator.predictWithLBP(textAnnotator.detectObjectsWithCanny(f_g), f_g, "object");

      result.insert(result.end(), objects.begin(), objects.end());

      return result;
  }

  vector<Annotation> annotateObjectsFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> face;
      vector<Annotation> dnn;
      vector<Annotation> contours;

      tbb::parallel_invoke(

        [&]
          () {
          dnn = caffeAnnotator.predictWithCAFFE(textAnnotator.detectObjectsWithCanny(f_g), f);
        },

        [&]
          () {
          face = faceAnnotator.predictWithLBP(f_g);
        },
        [&]
          () {
          contours = contoursAnnotator.detectContoursWithCanny(f_g);
        }
      );
      result.insert(result.end(), face.begin(), face.end());
      result.insert(result.end(), contours.begin(), contours.end());
      result.insert(result.end(), dnn.begin(), dnn.end());

      return result;
  }

  vector<Annotation> annotateTextsFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> texts;

      texts = textAnnotator.predictWithTESSERACT(f_g);
      result.insert(result.end(), texts.begin(), texts.end());

      return result;
  }

  vector<Annotation> annotateContoursFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> contours;

      contours = contoursAnnotator.detectContoursWithCanny(f_g);
      result.insert(result.end(), contours.begin(), contours.end());

      return result;
  }

  static void updateLBPModelFN(vector<cv::Mat> samples,
                               int label,
                               const string &ontology,
                               bool &aTInProgress) noexcept {
      cout << "Calling LBP model -> update\n";
      aTInProgress = true;
      objectsAnnotator.update(samples, label, ontology);
      aTInProgress = false;
  }

/**
* @function main
*/
  int main(int argc, char **argv) {

      textAnnotator.loadTESSERACTModel(".", "eng");

      String modelTxt = "bvlc_googlenet.prototxt";
      String modelBin = "bvlc_googlenet.caffemodel";

      faceAnnotator.loadCascadeClassifier(face_cascade_name);
      faceAnnotator.loadLBPModel(lbp_recognizer_name);

      caffeAnnotator.loadCAFFEModel(modelBin, modelTxt, "synset_words.txt");
      objectsAnnotator.loadLBPModel(lbp_recognizer_name);
      auto startRepl = [](std::size_t index, vector<shared_ptr<VisualREPL>> &cams) noexcept {
        if (cams[index]->startAt(static_cast<int>(index), 30)) {
            cout << "--(!)Camera found on " << index << " device index." << endl;
            return true;
        } else {
            cameras[index].reset();
            cerr << "--(!)Error opening video capture at: {" << index << "}\n You do have camera plugged in, right?"
                 << endl;
            return false;
        }
      };
      bool atLeastOneCamera = false;
      for (std::size_t i = 0; i < MAX_CAMERAS; i++) {
          std::string name{window_name + " Stream >> " + std::to_string(i)};
          switch (i) {
              case 0: {
                  cameras.push_back(make_shared<VisualREPL>(
                    VisualREPL(name + " [annotateObjectsFN]", clips, annotateObjectsFN,
                               updateLBPModelFN, WINDOW_SHOW)));
                  atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);
                  break;
              }
              case 1: {
                  cameras.push_back(make_shared<VisualREPL>(
                    VisualREPL(name + " [annotateFacesFN]", clips, annotateFacesFN,
                               updateLBPModelFN, WINDOW_SHOW)));
                  atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);
                  break;
              }
              case 2: {
                  cameras.push_back(make_shared<VisualREPL>(
                    VisualREPL(name + " [annotateTextsFN]", clips, annotateTextsFN,
                               updateLBPModelFN, WINDOW_SHOW)));
                  atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);

                  break;
              }
              case 3: {
                  cameras.push_back(make_shared<VisualREPL>(
                    VisualREPL(name + " [annotateContoursFN]", clips, annotateObjectsWithLBPFN,
                               updateLBPModelFN, WINDOW_SHOW)));
                  atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);

                  break;
              }
              default : {

              }
          }
      }
      if (!atLeastOneCamera) {
          cerr << "Can't find camera!Will std::terminate() everything!" << endl;
          std::terminate();
      }

      while (true) {
          this_thread::sleep_for(std::chrono::milliseconds(500));
          //-- bail out if escape was pressed
          DATA_OBJECT rv;
          clips.envEval("(facts)", rv);
          clips.envRun();
          clips.envReset();
      }
  }
}

int main(int argc, char **argv) {
    return homeaiapp::main(argc, argv);
}



