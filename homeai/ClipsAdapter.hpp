#pragma once

#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <chrono>
#include "opencv2/core.hpp"
#include "Annotation.hpp"

extern "C"
{
#include "clips/clips.h"
LOCALE void *EnvFindDeftemplate(void*, char*);
LOCALE struct fact* EnvCreateFact(void*, void*);
LOCALE void* EnvAddSymbol(void*, char*);
LOCALE void* EnvAssert(void*, void*);
LOCALE void* EnvCreateMultifield(void*, unsigned long);
LOCALE void* EnvAddLong(void*, long);

LOCALE intBool EnvAssignFactSlotDefaults(void*, void*);
LOCALE intBool EnvPutFactSlot(void*, void*, char*, DATA_OBJECT*);

LOCALE intBool DestroyEnvironment(void*);

LOCALE void* CreateEnvironment();
LOCALE int EnvBuild(void*, char*);
LOCALE int EnvLoad(void*, char*);
LOCALE void EnvReset(void*);
}

namespace hai {
    class ClipsAdapter {
    public:
        ClipsAdapter(const string aRulesFilePath) : rulesFilePath{aRulesFilePath} {
            theCLIPSEnv = CreateEnvironment();
            EnvBuild(theCLIPSEnv, const_cast<char*>(defaultDeftemplateFN().c_str()));
            EnvLoad(theCLIPSEnv, const_cast<char*>(aRulesFilePath.c_str()));
            EnvReset(theCLIPSEnv);
        }

        ~ClipsAdapter() { DestroyEnvironment(theCLIPSEnv); }

        inline void callFactCreateFN(Annotation& annotation, const string& visualRepl) noexcept {
            std::lock_guard<std::mutex> lock{createOneFactLock};
            addDetectFact2(theCLIPSEnv, annotation, visualRepl);
        }

        inline void callFactCreateFN(vector<Annotation>& annotations, const string& visualRepl) noexcept {
            std::lock_guard<std::mutex> lock{createFactsLock};
            for (auto& a : annotations) {
                addDetectFact2(theCLIPSEnv, a, visualRepl);
            }
        }

        inline void envReset() noexcept {
            std::lock_guard<std::mutex> lock{globalEnvLock};
            EnvReset(theCLIPSEnv);
        }

        inline void envRun() noexcept {
            std::lock_guard<std::mutex> lock{globalEnvLock};
            EnvRun(theCLIPSEnv, -1);
        }

        inline void envEval(string clipsCommand, DATA_OBJECT& result) noexcept {
            std::lock_guard<std::mutex> lock{globalEnvLock};
            EnvEval(theCLIPSEnv, const_cast<char*>(clipsCommand.c_str()), &result);
        }

        inline void envClear() noexcept {
        std::lock_guard<std::mutex> lock{globalEnvLock};
            EnvClear(theCLIPSEnv);
        }

    private:
        std::mutex createOneFactLock, createFactsLock, globalEnvLock;
        std::
        cv::Ptr<void> theCLIPSEnv;
        string rulesFilePath;

        string defaultDeftemplateFN(void) const noexcept {
            return "(deftemplate visualdetect"
                    " (slot type (default object))"
                    " (multislot rectangle)"
                    " (slot ontology)"
                    " (slot at)"
                    " (slot timestamp)"
                    " )";
        }

        void addDetectFact2(void* environment, Annotation& a, const string& visualRepl) noexcept {
            if (a.getType().compare("empty") == 0)
                return;
            void* newFact;
            void* templatePtr;
            void* theMultifield;
            DATA_OBJECT theValue;
            /*============================================================*/
            /* Disable garbage collection. It's only necessary to disable */
            /* garbage collection when calls are made into CLIPS from an */
            /* embedding program. It's not necessary to do this when the */
            /* the calls to user code are made by CLIPS (such as for */
            /* user-defined functions) or in the case of this example, */
            /* there are no calls to functions which can trigger garbage */
            /* collection (such as Send or FunctionCall). */
            /*============================================================*/
            //IncrementGCLocks(environment);
            /*==================*/
            /* Create the fact. */
            /*==================*/
            templatePtr = EnvFindDeftemplate(environment, const_cast<char*>("visualdetect"));
            newFact = EnvCreateFact(environment, templatePtr);
            if (newFact == NULL) return;
            /*==============================*/
            /* Set the value of the 'type' slot. */
            /*==============================*/
            theValue.type = SYMBOL;
            theValue.value = EnvAddSymbol(environment, const_cast<char*>(a.getType().c_str()));
            EnvPutFactSlot(environment, newFact, const_cast<char*>("type"), &theValue);
            /*==============================*/
            /* Set the value of the 'rectangle' slot. */
            /*==============================*/

            cv::Rect at;
            if (a.getType().compare("contour") == 0) {
                at = boundingRect(a.getContour());
            } else {
                at = a.getRectangle();
            }

            theMultifield = EnvCreateMultifield(environment, 4);
            SetMFType(theMultifield, 1, INTEGER);
            SetMFValue(theMultifield, 1, EnvAddLong(environment, at.x));

            SetMFType(theMultifield, 2, INTEGER);
            SetMFValue(theMultifield, 2, EnvAddLong(environment, at.y));

            SetMFType(theMultifield, 3, INTEGER);
            SetMFValue(theMultifield, 3, EnvAddLong(environment, at.width));

            SetMFType(theMultifield, 4, INTEGER);
            SetMFValue(theMultifield, 4, EnvAddLong(environment, at.height));


            SetDOBegin(theValue, 1);
            SetDOEnd(theValue, 4);
            theValue.type = MULTIFIELD;
            theValue.value = theMultifield;
            EnvPutFactSlot(environment, newFact, const_cast<char*>("rectangle"), &theValue);
            /*==============================*/
            /* Set the value of the what 'onthology' slot. */
            /*==============================*/
            theValue.type = SYMBOL;
            stringstream onto;
            onto << "\"" << a.getDescription() << "\"";
            theValue.value = EnvAddSymbol(environment, const_cast<char*>(onto.str().c_str()));
            EnvPutFactSlot(environment, newFact, const_cast<char*>("ontology"), &theValue);
            /*==============================*/
            /* Set the value of the what  'at' slot. */
            /*==============================*/
            theValue.type = SYMBOL;
            stringstream repl;
            repl << "\"" << visualRepl << "\"";
            theValue.value = EnvAddSymbol(environment, const_cast<char*>(repl.str().c_str()));
            EnvPutFactSlot(environment, newFact, const_cast<char*>("at"), &theValue);
            /*==============================*/
            /* Set the value of the what 'timestamp' slot. */
            /*==============================*/
            theValue.type = INTEGER;
            theValue.value = EnvAddLong(environment, std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            EnvPutFactSlot(environment, newFact, const_cast<char*>("timestamp"), &theValue);
            /*=================================*/
            /* Assign default values since all */
            /* slots were not initialized. */
            /*=================================*/
            EnvAssignFactSlotDefaults(environment, newFact);
            /*==========================================================*/
            /* Enable garbage collection. Each call to IncrementGCLocks */
            /* should have a corresponding call to DecrementGCLocks. */
            /*==========================================================*/
            //EnvDecrementGCLocks(environment);
            /*==================*/
            /* Assert the fact. */
            /*==================*/
            EnvAssert(environment, newFact);
        }
    };
}
