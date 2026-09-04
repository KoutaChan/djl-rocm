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
 * Contains backend-neutral APIs for differentiable fusion functions and bounded inference recipes.
 *
 * <p>{@link ai.djl.engine.fusion.FusionFunctions FusionFunctions} returns ordinary caller-owned
 * NDArrays and participates in the active engine's automatic differentiation graph. Engines may
 * implement these functions with fused forward and backward operations while preserving their
 * functional semantics. These results follow normal NDManager ownership and never refer to a
 * reusable session output slot.
 *
 * <p>The persistent recipe API is complementary and inference-only. A recipe declares a closed set
 * of stages rather than an arbitrary operator graph. Its {@link ai.djl.engine.fusion.FusionSession
 * FusionSession} reuses ring-slot output storage and engine-owned device-stream temporary arenas,
 * so session submissions do not participate in automatic differentiation. Use a corresponding
 * method in {@code FusionFunctions} when gradients are required.
 *
 * <p>{@link ai.djl.engine.fusion.FusionRecipe.AffineSum AffineSum} projects and sums dynamic
 * values, fixed singleton-leading values, and an optional bias before applying a supported
 * activation. A fixed value shaped {@code [1, ..., featureWidth]} broadcasts over the active
 * leading extent; each source may independently use FLOAT16, BFLOAT16, or FLOAT32 while weights
 * select the common projection and output data type. Backends may convert and project constant
 * fixed values once when constants are bound. {@link ai.djl.engine.fusion.FusionRecipe.OutputPack
 * OutputPack} writes several score values into one persistent FLOAT32 output. {@link
 * ai.djl.engine.fusion.FusionRecipe.ProjectedResidualMlp ProjectedResidualMlp} evaluates a
 * two-projection SiLU MLP whose first projection supplies both its residual and hidden branches.
 * The input's fixed prefix dimensions are preserved. {@link
 * ai.djl.engine.fusion.FusionRecipe.IndexedAffine IndexedAffine} gathers selected rows from mixed
 * floating-point sources, evaluates a fixed two-layer projection, and scatters the results into a
 * zero-filled dense output without materializing individual gather or concatenation values. {@link
 * ai.djl.engine.fusion.FusionRecipe.TransformerEncoderStack TransformerEncoderStack} declares one
 * or more short, dense pre-normalized attention blocks with an affine output normalization. It
 * keeps each block's unnormalized attention residual available to the feed-forward branch, while
 * allowing a backend to combine attention, projection, residual, activation, and normalization
 * boundaries into a bounded set of native launches. Projection parameters use the stack data type;
 * normalization parameters may additionally remain FLOAT32 for mixed-precision inference. {@link
 * ai.djl.engine.fusion.FusionRecipe.MappedGroupedMaskedSoftmaxPoolGroup
 * MappedGroupedMaskedSoftmaxPoolGroup} applies several masked softmax pools to one candidate memory
 * and writes multiple destination-mapped context and presence sets in one stage. Each set remains
 * contiguous, repeated source groups are allowed, and a {@code -1} destination is zero.
 *
 * <p>The persistent inference lifecycle is {@code recipe -> plan -> executable -> session ->
 * invocation -> output lease}. Preparation validates shapes and builds a bounded command plan.
 * Binding retains caller constants and may create backend-owned packed constants or precomputed
 * values. Preparation may compile smaller storage-capacity profiles in addition to the always
 * available recipe-maximum plan. A session fixes one selected profile for its lifetime and owns a
 * ring of output slots at that capacity. The engine shares a planner arena across sessions that
 * submit on the same device stream. One invocation submits the whole recipe, and its lease keeps
 * the selected output slot alive until downstream work no longer uses its outputs. {@link
 * ai.djl.engine.fusion.FusionCompilationReport FusionCompilationReport} reports shared
 * per-executable storage, session-retained slot storage, and execution-lane arena capacity
 * separately for the maximum plan and each compiled profile.
 *
 * <p>Execution sessions are externally serialized: method executions using a session and its
 * derived handles are not thread-safe and must not overlap. Outstanding handle lifetimes may
 * coexist on distinct ring slots. The first submission selects the session's accelerator stream;
 * sequential calls may move between threads only while that same stream is current.
 */
package ai.djl.engine.fusion;
