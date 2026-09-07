/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dlfcn.h>
#include <jni.h>

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_RocmLibraryLoader_load(
    JNIEnv* env, jclass loader, jstring path) {
  (void)loader;
  const char* filename = (*env)->GetStringUTFChars(env, path, NULL);
  if (filename == NULL) {
    return;
  }

  // Preserve the private symlink path as ELF $ORIGIN. System.load resolves it
  // to the original SDK/LibTorch directory before calling the dynamic linker.
  // Dependencies remain loaded for the lifetime of the engine process.
  if (dlopen(filename, RTLD_LAZY | RTLD_GLOBAL) == NULL) {
    const char* error = dlerror();
    jclass exception = (*env)->FindClass(env, "java/lang/UnsatisfiedLinkError");
    if (exception != NULL) {
      (*env)->ThrowNew(env, exception, error);
    }
  }
  (*env)->ReleaseStringUTFChars(env, path, filename);
}
