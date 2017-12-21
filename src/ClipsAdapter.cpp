#include "../headers/ClipsAdapter.hpp"
namespace hai {
  ClipsAdapter::ClipsAdapter(const std::string aRulesFilePath) : rulesFilePath{aRulesFilePath} {
      theCLIPSEnv = CreateEnvironment();
      EnvBuild(theCLIPSEnv, const_cast<char *>(defaultDeftemplateFN().c_str()));
      EnvLoad(theCLIPSEnv, const_cast<char *>(aRulesFilePath.c_str()));
      EnvReset(theCLIPSEnv);
  }

  ClipsAdapter::~ClipsAdapter() { DestroyEnvironment(theCLIPSEnv); }

  void ClipsAdapter::callFactCreateFN(Annotation &annotation, const std::string &visualRepl) noexcept {
      tbb::mutex::scoped_lock lock{ClipsAdapter::createOneFactLock};
      addDetectFact2(theCLIPSEnv, annotation, visualRepl);
  }
  void ClipsAdapter::callFactCreateFN(std::vector<Annotation> &annotations,
                                      const std::string &visualRepl) noexcept {
      tbb::mutex::scoped_lock lock{createFactsLock};
      for (auto &a : annotations) {
          addDetectFact2(theCLIPSEnv, a, visualRepl);
      }
  }
  void ClipsAdapter::envReset() noexcept {
      tbb::mutex::scoped_lock lock{globalEnvLock};
      EnvReset(theCLIPSEnv);
  }
  void ClipsAdapter::envRun() noexcept {
      tbb::mutex::scoped_lock lock{globalEnvLock};
      EnvRun(theCLIPSEnv, -1);
  }
  void ClipsAdapter::envEval(std::string clipsCommand, DATA_OBJECT &result) noexcept {
      tbb::mutex::scoped_lock lock{globalEnvLock};
      EnvEval(theCLIPSEnv, const_cast<char *>(clipsCommand.c_str()), &result);
  }
  void ClipsAdapter::envClear() noexcept {
      tbb::mutex::scoped_lock lock{globalEnvLock};
      EnvClear(theCLIPSEnv);
  }

  void ClipsAdapter::addDetectFact2(void *environment, Annotation &a, const std::string &visualRepl) noexcept {
      if (a.getType().compare("empty")==0)
          return;
      void *newFact;
      void *templatePtr;
      void *theMultifield;
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
      templatePtr = EnvFindDeftemplate(environment, const_cast<char *>("visualdetect"));
      newFact = EnvCreateFact(environment, templatePtr);
      if (newFact==nullptr) return;
      /*==============================*/
      /* Set the value of the 'type' slot. */
      /*==============================*/
      theValue.type = SYMBOL;
      theValue.value = EnvAddSymbol(environment, const_cast<char *>(a.getType().c_str()));
      EnvPutFactSlot(environment, newFact, const_cast<char *>("type"), &theValue);
      /*==============================*/
      /* Set the value of the 'rectangle' slot. */
      /*==============================*/

      cv::Rect at;
      if (a.getType().compare("contour")==0) {
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
      EnvPutFactSlot(environment, newFact, const_cast<char *>("rectangle"), &theValue);
      /*==============================*/
      /* Set the value of the what 'onthology' slot. */
      /*==============================*/
      theValue.type = SYMBOL;
      std::stringstream onto;
      onto << "\"" << a.getDescription() << "\"";
      theValue.value = EnvAddSymbol(environment, const_cast<char *>(onto.str().c_str()));
      EnvPutFactSlot(environment, newFact, const_cast<char *>("ontology"), &theValue);
      /*==============================*/
      /* Set the value of the what  'at' slot. */
      /*==============================*/
      theValue.type = SYMBOL;
      std::stringstream repl;
      repl << "\"" << visualRepl << "\"";
      theValue.value = EnvAddSymbol(environment, const_cast<char *>(repl.str().c_str()));
      EnvPutFactSlot(environment, newFact, const_cast<char *>("at"), &theValue);
      /*==============================*/
      /* Set the value of the what 'timestamp' slot. */
      /*==============================*/
      theValue.type = INTEGER;
      theValue.value = EnvAddLong(environment, std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
      EnvPutFactSlot(environment, newFact, const_cast<char *>("timestamp"), &theValue);
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

  std::string ClipsAdapter::defaultDeftemplateFN(void) const noexcept {
      return std::string{"(deftemplate visualdetect"
                           " (slot type (default object))"
                           " (multislot rectangle)"
                           " (slot ontology)"
                           " (slot at)"
                           " (slot timestamp)"
                           " )"};
  }

  tbb::mutex ClipsAdapter::createOneFactLock;
  tbb::mutex ClipsAdapter::createFactsLock;
  tbb::mutex ClipsAdapter::globalEnvLock;
}
