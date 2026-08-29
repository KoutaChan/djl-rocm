/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
#include <torch/torch.h>
// clang-format off
#ifdef V1_10_X
    #include <torch/csrc/jit/frontend/code_template.h>
#else
    #include <ATen/code_template.h>
#endif
#include <ATen/core/jit_type.h>
// clang-format on

#include <sstream>

#include <djl/utils.h>
#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_accelerator.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"
#include "ai_djl_pytorch_jni_cache.h"

#if defined(__ANDROID__)
#ifndef USE_PTHREADPOOL
#define USE_PTHREADPOOL
#endif /* USE_PTHREADPOOL */
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif

using namespace torch::autograd::profiler;

// The file is the implementation for PyTorch system-wide operations

struct ThreadInferenceMode {
#if defined(V1_10_X)
  torch::NoGradGuard guard;
#else
  c10::InferenceMode guard;
#endif
};

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIsGradMode(JNIEnv* env, jobject jthis) {
  API_BEGIN()
#if defined(__ANDROID__)
  return false;
#else
  return c10::GradMode::is_enabled();
#endif
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSetGradMode(
    JNIEnv* env, jobject jthis, jboolean enable) {
  API_BEGIN()
#if !defined(__ANDROID__)
  c10::GradMode::set_enabled(enable);
#endif
  API_END()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetNumInteropThreads(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return torch::get_num_interop_threads();
  API_END_RETURN()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetNumThreads(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return torch::get_num_threads();
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSetNumInteropThreads(
    JNIEnv* env, jobject jthis, jint jthreads) {
  API_BEGIN()
#if defined(__ANDROID__)
  Log log(env);
  log.info("Android didn't support this interop config, please use intra-op instead");
#else
  torch::set_num_interop_threads(jthreads);
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSetNumThreads(
    JNIEnv* env, jobject jthis, jint jthreads) {
  API_BEGIN()
#if defined(__ANDROID__)
  caffe2::pthreadpool()->set_thread_count(jthreads);
#else
  torch::set_num_threads(jthreads);
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchManualSeed(JNIEnv* env, jobject jthis, jlong jseed) {
  API_BEGIN()
  torch::manual_seed(jseed);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchShowConfig(
    JNIEnv* env, jobject jthis, jobject jset) {
  API_BEGIN()
  jclass set_class = env->GetObjectClass(jset);
  if (set_class == nullptr) {
    env->ThrowNew(NULL_PTR_EXCEPTION_CLASS, "Java Set class is not found");
    return;
  }
  jmethodID add_method_id = env->GetMethodID(set_class, "add", "(Ljava/lang/Object;)Z");
  if (add_method_id == nullptr) {
    env->ThrowNew(NULL_PTR_EXCEPTION_CLASS, "The add method in Set is not found");
    return;
  }
  std::string feature;
  jstring jfeature;
#if !defined(__ANDROID__)
  if (torch::cuda::is_available()) {
    feature = "CUDA";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::cuda::cudnn_is_available()) {
    feature = "CUDNN";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
#ifdef USE_DISTRIBUTED_NCCL
  feature = "NATIVE_DISTRIBUTED_NCCL";
  jfeature = env->NewStringUTF(feature.c_str());
  env->CallBooleanMethod(jset, add_method_id, jfeature);
  env->DeleteLocalRef(jfeature);
#endif
#endif
  if (torch::hasMKL()) {
    feature = "MKL";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::hasMKLDNN()) {
    feature = "MKLDNN";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::hasOpenMP()) {
    feature = "OPENMP";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOpenInferenceMode(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return reinterpret_cast<uintptr_t>(new ThreadInferenceMode());
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCloseInferenceMode(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  delete reinterpret_cast<ThreadInferenceMode*>(jhandle);
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOpenStreamScope(
    JNIEnv* env, jobject jthis, jintArray jdevice) {
  API_BEGIN()
  const torch::Device device = utils::GetDeviceFromJDevice(env, jdevice);
  return reinterpret_cast<uintptr_t>(djl_pytorch::accel::NewStreamScope(device));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCloseStreamScope(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::DeleteStreamScope(
      reinterpret_cast<djl_pytorch::accel::StreamScope*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateDeviceStream(
    JNIEnv* env, jobject jthis, jintArray jdevice) {
  API_BEGIN()
  const torch::Device device = utils::GetDeviceFromJDevice(env, jdevice);
  return reinterpret_cast<uintptr_t>(djl_pytorch::accel::NewDeviceStream(device));
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOpenDeviceStream(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* stream = reinterpret_cast<djl_pytorch::accel::DeviceStream*>(jhandle);
  return reinterpret_cast<uintptr_t>(djl_pytorch::accel::OpenDeviceStream(stream));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteDeviceStream(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::DeleteDeviceStream(
      reinterpret_cast<djl_pytorch::accel::DeviceStream*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateDeviceEvent(
    JNIEnv* env, jobject jthis, jintArray jdevice) {
  API_BEGIN()
  const torch::Device device = utils::GetDeviceFromJDevice(env, jdevice);
  return reinterpret_cast<uintptr_t>(djl_pytorch::accel::NewDeviceEvent(device));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRecordDeviceEvent(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::RecordDeviceEvent(
      reinterpret_cast<djl_pytorch::accel::DeviceEvent*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchWaitDeviceEvent(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::WaitDeviceEvent(
      reinterpret_cast<djl_pytorch::accel::DeviceEvent*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchQueryDeviceEvent(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return djl_pytorch::accel::QueryDeviceEvent(
      reinterpret_cast<djl_pytorch::accel::DeviceEvent*>(jhandle));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSynchronizeDeviceEvent(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::SynchronizeDeviceEvent(
      reinterpret_cast<djl_pytorch::accel::DeviceEvent*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteDeviceEvent(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::DeleteDeviceEvent(
      reinterpret_cast<djl_pytorch::accel::DeviceEvent*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateAcceleratorGraph(
    JNIEnv* env, jobject jthis, jintArray jdevice) {
  API_BEGIN()
  const torch::Device device = utils::GetDeviceFromJDevice(env, jdevice);
  return reinterpret_cast<uintptr_t>(djl_pytorch::accel::NewAcceleratorGraph(device));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchBeginAcceleratorGraphCapture(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::BeginAcceleratorGraphCapture(
      reinterpret_cast<djl_pytorch::accel::AcceleratorGraph*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchEndAcceleratorGraphCapture(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::EndAcceleratorGraphCapture(
      reinterpret_cast<djl_pytorch::accel::AcceleratorGraph*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchReplayAcceleratorGraph(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::ReplayAcceleratorGraph(
      reinterpret_cast<djl_pytorch::accel::AcceleratorGraph*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteAcceleratorGraph(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl_pytorch::accel::DeleteAcceleratorGraph(
      reinterpret_cast<djl_pytorch::accel::AcceleratorGraph*>(jhandle));
  API_END()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetGpuCount(JNIEnv* env, jobject jthis) {
  API_BEGIN()
#if defined(__ANDROID__)
  return 0;
#else
  return static_cast<jint>(torch::cuda::device_count());
#endif
  API_END_RETURN()
}

std::string ToString(const std::vector<std::vector<int64_t>>& shapes) {
  std::ostringstream oss;
  oss << "[";
  for (auto i = 0; i < shapes.size(); ++i) {
    oss << "[";
    if (!shapes[i].empty()) {
      std::copy(shapes[i].begin(), shapes[i].end() - 1, std::ostream_iterator<int64_t>(oss, ", "));
      oss << shapes[i].back();
    }
    if (i == shapes.size() - 1) {
      oss << "]";
    } else {
      oss << "], ";
    }
  }
  oss << "]";
  return oss.str();
}

inline std::string FormatMemory(int64_t bytes) {
  int64_t kb = 1024;
  int64_t mb = 1024 * 1024;
  int64_t gb = 1024 * 1024 * 1024;
  std::ostringstream oss;
  oss.precision(2);
  if (std::abs(bytes) >= gb) {
    oss << std::fixed << static_cast<double>(bytes) / gb << " Gb";
  } else if (std::abs(bytes) >= mb) {
    oss << std::fixed << static_cast<double>(bytes) / mb << " Mb";
  } else if (std::abs(bytes) >= kb) {
    oss << std::fixed << static_cast<double>(bytes) / kb << " Kb";
  } else {
    oss << bytes << " b";
  }
  return oss.str();
}

// the code snippet is copied from torch/csrc/autograd/profiler_legacy.cpp
#ifdef V1_10_X
static torch::jit::CodeTemplate event_template(R"(
{
  "name": "${name}",
  "ph": "X",
  "ts": ${ts},
  "dur": ${dur},
  "tid": ${tid},
  "pid": "CPU Functions",
  "shape": ${shape},
  "cpu mem": "${cpu_mem}",
  "gpu dur": ${gpu_dur},
  "device": ${device},
  "args": {}
})");
#else
static const at::jit::CodeTemplate event_template(R"(
{
  "name": "${name}",
  "ph": "X",
  "ts": ${ts},
  "dur": ${dur},
  "tid": ${tid},
  "pid": "CPU Functions",
  "shape": ${shape},
  "cpu mem": "${cpu_mem}",
  "gpu dur": ${gpu_dur},
  "device": ${device},
  "args": {}
})");
#endif

void WriteProfilerEventsToStream(std::ostream& out, const std::vector<std::vector<LegacyEvent*>>& thread_events) {
  TORCH_CHECK(out, "Could not open file");
  std::set<std::string> filtered_out_names = {
      "profiler::_record_function_enter", "profiler::_record_function_exit", "is_leaf", "output_nr", "_version"};
  LegacyEvent* profiler_start = nullptr;
  for (const auto& events : thread_events) {
    for (auto e : events) {
      if (0 == strcmp(e->name(), "__start_profile")) {
        profiler_start = e;
        break;
      }
    }
  }
  TORCH_CHECK(profiler_start, "Could not find __start_profile mark");

  struct PairHash {
    size_t operator()(std::pair<at::RecordFunctionHandle, int> p) const noexcept {
      return std::hash<at::RecordFunctionHandle>()(p.first) ^ std::hash<int64_t>()(p.second);
    }
  };
  out << "[\n";
  bool first = true;
  for (const std::vector<LegacyEvent*>& thread_event_list : thread_events) {
    // accumulated memory allocations per handle
    std::unordered_map<std::pair<at::RecordFunctionHandle, int64_t>, int64_t, PairHash> cpu_memory_allocs;
    std::unordered_map<std::pair<at::RecordFunctionHandle, int64_t>, LegacyEvent*, PairHash> events_map;
    std::set<std::pair<at::RecordFunctionHandle, int64_t>> filtered_handles;
    for (LegacyEvent* evt : thread_event_list) {
      auto event_key = std::make_pair<at::RecordFunctionHandle, int64_t>(evt->handle(), evt->nodeId());
      if (filtered_out_names.find(evt->name()) != filtered_out_names.end() ||
          filtered_handles.find(event_key) != filtered_handles.end()) {
        filtered_handles.insert(event_key);
        continue;
      }
      if (evt->kindStr() == "push") {
        events_map[event_key] = evt;
        cpu_memory_allocs[event_key] = 0;
      } else if (evt->kindStr() == "pop") {
        if (!first) {
          out << ",\n";
        }
        first = false;
        auto it = events_map.find(event_key);
        auto mem_it = cpu_memory_allocs.find(event_key);
        TORCH_CHECK(it != events_map.end(), "Unmatched pop event");
        LegacyEvent* start = it->second;
        int64_t memory_usage = mem_it->second;

#ifdef V1_10_X
        torch::jit::TemplateEnv env;
#else
        at::jit::TemplateEnv env;
#endif
        env.s("name", start->name());
        env.d("ts", profiler_start->cpuElapsedUs(*start));
        env.d("dur", start->cpuElapsedUs(*evt));
        env.d("tid", start->threadId());
        // we add extra info here
        env.s("shape", ToString(start->shapes()));
        env.s("cpu_mem", FormatMemory(memory_usage));
        bool has_gpu_timing = start->hasCuda() && evt->hasCuda();
        env.d("gpu_dur", has_gpu_timing ? start->cudaElapsedUs(*evt) : -1.0);
        env.d("device", has_gpu_timing ? start->device() : -1);
        out << event_template.format(env);

        events_map.erase(it);
        cpu_memory_allocs.erase(mem_it);
      } else if (evt->kindStr() == "memory_alloc") {
        for (const auto& e : cpu_memory_allocs) {
          cpu_memory_allocs[e.first] += evt->cpuMemoryUsage();
        }
      }
    }
  }
  out << "]\n";
}
// end of copies

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchStartProfile(
    JNIEnv* env, jobject jthis, jboolean juse_cuda, jboolean jrecord_shape, jboolean jprofile_memory) {
  API_BEGIN()
  if (profilerEnabled()) {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "please call stopProfile before you start a new section");
    return;
  }
  enableProfilerLegacy(ProfilerConfig(juse_cuda ? ProfilerState::CUDA : ProfilerState::CPU,
      /* report_input_shapes */ jrecord_shape,
      /* profile_memory */ jprofile_memory));
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchStopProfile(
    JNIEnv* env, jobject jthis, jstring joutput_file) {
  API_BEGIN()
  if (!profilerEnabled()) {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "please call startProfiler() before you use stopProfile!");
    return;
  }
  std::string output_file = djl::utils::jni::GetStringFromJString(env, joutput_file);
  std::ofstream file(output_file);
  std::vector<std::vector<LegacyEvent>> event_lists = disableProfilerLegacy();
  std::vector<std::vector<LegacyEvent*>> event_ptr_lists;
  for (auto& l : event_lists) {
    std::vector<LegacyEvent*> events;
    for (auto& e : l) {
      events.emplace_back(&e);
    }
    event_ptr_lists.emplace_back(events);
  }
  WriteProfilerEventsToStream(file, event_ptr_lists);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCudaEmptyCache(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  djl_pytorch::accel::EmptyCache();
  API_END()
}
