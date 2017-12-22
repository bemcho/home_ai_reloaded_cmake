#include "tbb/parallel_invoke.h"
#include "../headers/Annotation.hpp"
#include "../headers/VisualContextAnnotator.hpp"
#include "../headers/ClipsAdapter.hpp"
#include "../headers/VisualREPL.hpp"

namespace homeaiapp {

  using namespace hai;

  const std::string window_name{"Home AI"};
  const std::string face_cascade_name{"./face/cascade_frontalface.xml"};
  const std::string lbp_face_recognizer_name{"./face/lbphFaceRecognizer.xml"};
  const std::string lbp_object_recognizer_name{"./object/lbphObjectRecognizer.xml"};
  const std::string clips_vca_rules{"./clips/visualcontextrules.clp"};
  const std::string caffe_prototxt_file = "./caffe/bvlc_googlenet.prototxt";
  const std::string caffe_model_file = "./caffe/bvlc_googlenet.caffemodel";
  const std::string caffe_synset_words_file = "./caffe/synset_words.txt";

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

  vector<Annotation> annotateObjectsWithCaffeFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> dnn;

      dnn =  caffeAnnotator.predictWithCAFFE(textAnnotator.detectObjectsWithCanny(f_g), f);

      result.insert(result.end(), dnn.begin(), dnn.end());

      return result;
  }

  vector<Annotation> annotateObjectsFN(const Mat &f, const Mat &f_g) noexcept {
      vector<Annotation> result;
      vector<Annotation> face;
      vector<Annotation> objects;
      vector<Annotation> contours;

      tbb::parallel_invoke(

        [&]
          () {
          objects = objectsAnnotator.predictWithLBP(textAnnotator.detectObjectsWithCanny(f_g), f_g, "object");
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
      result.insert(result.end(), objects.begin(), objects.end());

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

      faceAnnotator.loadCascadeClassifier(face_cascade_name);
      faceAnnotator.loadLBPModel(lbp_face_recognizer_name);

      caffeAnnotator.loadCAFFEModel(caffe_model_file, caffe_prototxt_file, caffe_synset_words_file);
      objectsAnnotator.loadLBPModel(lbp_object_recognizer_name);

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
                    VisualREPL(name + " [annotateFacesFN]", clips, annotateObjectsWithCaffeFN,
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
                    VisualREPL(name + " [annotateContoursFN]", clips, annotateFacesFN,
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
          this_thread::sleep_for(std::chrono::milliseconds(1000));
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



