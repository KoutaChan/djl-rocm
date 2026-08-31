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
 * <p>A recipe declares a closed set of inference stages rather than an arbitrary operator graph.
 * {@link ai.djl.engine.fusion.FusionRecipe.AffineSum AffineSum} projects and sums dynamic values,
 * fixed singleton-leading values, and an optional bias before applying a supported activation. A
 * fixed value shaped {@code [1, ..., featureWidth]} broadcasts over the active leading extent; each
 * source may independently use FLOAT16, BFLOAT16, or FLOAT32 while weights select the common
 * projection and output data type. Backends may convert and project constant fixed values once when
 * constants are bound. {@link ai.djl.engine.fusion.FusionRecipe.OutputPack OutputPack} writes
 * several score values into one persistent FLOAT32 output. {@link
 * ai.djl.engine.fusion.FusionRecipe.IndexedAffine IndexedAffine} gathers selected rows from mixed
 * floating-point sources, evaluates a fixed two-layer projection, and scatters the results into a
 * zero-filled dense output without materializing individual gather or concatenation values. {@link
 * ai.djl.engine.fusion.FusionRecipe.TransformerEncoderStack TransformerEncoderStack} declares one
 * or more short, dense pre-normalized attention blocks with an affine output normalization. It
 * keeps each block's unnormalized attention residual available to the feed-forward branch, while
 * allowing a backend to combine attention, projection, residual, activation, and normalization
 * boundaries into a bounded set of native launches. Projection parameters use the stack data type;
 * normalization parameters may additionally remain FLOAT32 for mixed-precision inference.
 *
 * <p>The lifecycle is {@code recipe -> plan -> executable -> session -> invocation -> output
 * lease}. Preparation validates shapes and builds a bounded command plan. Binding retains caller
 * constants and may create backend-owned packed constants or precomputed values. A session owns a
 * ring of maximum-shape output and workspace slots. One invocation submits the whole recipe, and
 * its lease keeps the selected slot alive until downstream work no longer uses its outputs. {@link
 * ai.djl.engine.fusion.FusionCompilationReport FusionCompilationReport} reports shared
 * per-executable storage separately from output and workspace storage allocated for every slot.
 *
 * <p>Execution sessions are externally serialized: method executions using a session and its
 * derived handles are not thread-safe and must not overlap. Outstanding handle lifetimes may
 * coexist on distinct ring slots, and sequential calls may move between threads.
 */
package ai.djl.engine.fusion;
