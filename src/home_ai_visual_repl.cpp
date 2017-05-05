#include "tbb/parallel_invoke.h"
#include "homeai/Annotation.hpp"
#include "homeai/VisualContextAnnotator.hpp"
#include "homeai/ClipsAdapter.hpp"
#include "homeai/VisualREPL.hpp"

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

    std::size_t MAX_CAMERAS = 5;
    bool WINDOW_SHOW = true;

    vector<Annotation> annotateFaceContoursFN(Mat f, Mat f_g) {
        vector<Annotation> result;
        vector<Annotation> face;
        vector<Annotation> contours;
        vector<Annotation> objects;

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
                    contours = objectsAnnotator.detectContoursWithCanny(f_g);
                }
        );
        result.insert(result.end(), face.begin(), face.end());
        result.insert(result.end(), objects.begin(), objects.end());
        result.insert(result.end(), contours.begin(), contours.end());

        return result;
    }

    vector<Annotation> annotateObjectsFN(Mat f, Mat f_g) {
        vector<Annotation> result;
        vector<Annotation> face;
        vector<Annotation> objects;
        vector<Annotation> contours;
        vector<Annotation> texts;
       // vector<Annotation> dnn;

        tbb::parallel_invoke(
                [&]
                        () {
                    texts = textAnnotator.predictWithTESSERACT(f_g);
                },
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
                    contours = objectsAnnotator.detectContoursWithCanny(f_g);
                }
        );
        result.insert(result.end(), texts.begin(), texts.end());
        result.insert(result.end(), objects.begin(), objects.end());
        result.insert(result.end(), face.begin(), face.end());
        result.insert(result.end(), contours.begin(), contours.end());
        //result.insert(result.end(), dnn.begin(), dnn.end());

        return result;
    }

   vector<Annotation> annotateTextContoursFN(Mat f, Mat f_g) {
        vector<Annotation> result;
        vector<Annotation> objects;
        vector<Annotation> contours;

        tbb::parallel_invoke(
                [&]
                        () {
                    objects = textAnnotator.predictWithTESSERACT(f_g);

                },
                [&]
                        () {
                    contours = objectsAnnotator.detectContoursWithCanny(f_g);
                }
        );
        result.insert(result.end(), objects.begin(), objects.end());
        result.insert(result.end(), contours.begin(), contours.end());

        return result;
    }

    static void updateLBPModelFN(vector<cv::Mat> samples, int label, string ontology, bool& aTInProgress) {
        tbb::mutex::scoped_lock(trainMutex);
        cout << "Calling LBP model -> update\n";
        aTInProgress = true;
        objectsAnnotator.update(samples, label, ontology);
        aTInProgress = false;
    }

/**
* @function main
*/
    int main(int argc, char** argv) {

        textAnnotator.loadTESSERACTModel(".", "eng");

        String modelTxt = "bvlc_googlenet.prototxt";
        String modelBin = "bvlc_googlenet.caffemodel";

        faceAnnotator.loadCascadeClassifier(face_cascade_name);
        faceAnnotator.loadLBPModel(lbp_recognizer_name);

        objectsAnnotator.loadCAFFEModel(modelBin, modelTxt, "synset_words.txt");
        objectsAnnotator.loadLBPModel(lbp_recognizer_name);
        auto startRepl = [](std::size_t index, vector<shared_ptr<VisualREPL>>& cams) {
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
            switch (i) {
                case 0: {
                    cameras.push_back(make_shared<VisualREPL>(
                            VisualREPL(window_name + " Stream " + std::to_string(i), clips, annotateObjectsFN,
                                       updateLBPModelFN, WINDOW_SHOW)));
                    atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);
                    break;
                }
                case 1: {
                    cameras.push_back(make_shared<VisualREPL>(
                            VisualREPL(window_name + " Stream " + std::to_string(i), clips, annotateFaceContoursFN,
                                       updateLBPModelFN, WINDOW_SHOW)));
                    atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);
                    break;
                }
                case 2: {
                    cameras.push_back(make_shared<VisualREPL>(
                            VisualREPL(window_name + " Stream " + std::to_string(i), clips, annotateTextContoursFN,
                                       updateLBPModelFN, WINDOW_SHOW)));
                    atLeastOneCamera = atLeastOneCamera || startRepl(i, cameras);

                    break;
                }
                case 3: {
                    cameras.push_back(make_shared<VisualREPL>(
                            VisualREPL(window_name + " Stream " + std::to_string(i), clips, annotateFaceContoursFN,
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

            this_thread::sleep_for(std::chrono::milliseconds(5));
            //DATA_OBJECT rv;
            //clips.envEval("(facts)", rv);
            clips.envRun();
            this_thread::sleep_for(std::chrono::milliseconds(5));
            //-- bail out if escape was pressed
            if (waitKey(1) == 27) {
                break;
            }

        }
        std::terminate();

        return EXIT_SUCCESS;
    }
}

int main(int argc, char** argv) {
    return homeaiapp::main(argc, argv);
}



