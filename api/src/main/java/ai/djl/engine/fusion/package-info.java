/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/**
 * Contains backend-neutral APIs for preparing and repeatedly executing bounded fusion recipes.
 *
 * <p>Execution sessions are externally serialized: method executions using a session and its
 * derived handles are not thread-safe and must not overlap. Outstanding handle lifetimes may
 * coexist on distinct ring slots, and sequential calls may move between threads.
 */
package ai.djl.engine.fusion;
