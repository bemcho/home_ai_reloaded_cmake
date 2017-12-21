#pragma once

#include <string>
#include <functional>
#include <memory>
#include "tbb/mutex.h"
#include <chrono>
#include "opencv2/imgproc.hpp"
#include "Annotation.hpp"

extern "C"
{
#include "clips/clips.h"
LOCALE void *EnvFindDeftemplate(void *, char *);
LOCALE struct fact *EnvCreateFact(void *, void *);
LOCALE void *EnvAddSymbol(void *, char *);
LOCALE void *EnvAssert(void *, void *);
LOCALE void *EnvCreateMultifield(void *, unsigned long);
LOCALE void *EnvAddLong(void *, long);

LOCALE intBool EnvAssignFactSlotDefaults(void *, void *);
LOCALE intBool EnvPutFactSlot(void *, void *, char *, DATA_OBJECT *);

LOCALE intBool DestroyEnvironment(void *);

LOCALE void *CreateEnvironment();
LOCALE int EnvBuild(void *, char *);
LOCALE int EnvLoad(void *, char *);
LOCALE void EnvReset(void *);
}

namespace hai {
  class ClipsAdapter {
   public:
      ClipsAdapter(const std::string aRulesFilePath);

      ~ClipsAdapter();

      void callFactCreateFN(Annotation &annotation, const std::string &visualRepl) noexcept;

      void callFactCreateFN(std::vector<Annotation> &annotations, const std::string &visualRepl) noexcept;

      void envReset() noexcept;

      void envRun() noexcept;

      void envEval(std::string clipsCommand, DATA_OBJECT &result) noexcept;

      void envClear() noexcept;

   private:
      std::string defaultDeftemplateFN(void) const noexcept;
      static tbb::mutex createOneFactLock, createFactsLock, globalEnvLock;
      cv::Ptr<void> theCLIPSEnv;
      std::string rulesFilePath;
      void addDetectFact2(void *environment, Annotation &a, const std::string &visualRepl) noexcept;
  };
}
