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
package ai.djl.engine.fusion;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Describes a bounded, inference-only computation that an engine can prepare for repeated use.
 *
 * <p>A recipe is immutable and does not contain engine resources. Tensor storage has a fixed
 * maximum shape. A tensor may use one named leading {@link Dimension}; an invocation supplies the
 * active extent of that dimension without changing the storage capacity. Inner dimensions are
 * always fixed.
 *
 * <p>The API exposes a closed set of inference stages. {@link AffineSum} projects several values
 * into one width, adds the projected values with fixed-shape broadcasting, and optionally applies
 * an activation. {@link IndexedAffine} gathers selected rows from several values, evaluates a
 * bounded two-layer projection, and scatters the selected results into a dense value. {@link
 * OutputPack} concatenates two-dimensional floating-point values along their last axis and converts
 * them into a contiguous {@link DataType#FLOAT32} value. {@link BinaryBranchBlend} selects or
 * blends two branch contexts from their presence values and a binary logit. {@link
 * TransformerEncoderStack} executes one or more fixed-width, pre-normalized transformer encoder
 * blocks over a short dense sequence. {@link SingleQueryCrossAttentionReadoutGroup} evaluates
 * several purpose-specific single-query readouts over one shared masked memory without
 * materializing projected keys and values. {@link IndexedLocalTransformerEncoder} evaluates one
 * transformer block over compact rows selected from fixed-size local groups. Additional value types
 * can be added without changing the lifecycle of prepared plans and sessions.
 */
public final class FusionRecipe {

    private final String name;
    private final List<Dimension> dimensions;
    private final List<Input> inputs;
    private final List<Constant> constants;
    private final List<Value> values;
    private final List<Output> outputs;

    private FusionRecipe(Builder builder) {
        name = builder.name;
        dimensions = immutableCopy(builder.dimensions);
        inputs = immutableCopy(builder.inputs);
        constants = immutableCopy(builder.constants);
        values = immutableCopy(builder.values);
        outputs = immutableCopy(builder.outputs);
    }

    /**
     * Creates a builder for a fusion recipe.
     *
     * @param name the diagnostic name of the recipe
     * @return a new builder
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Returns the diagnostic name of this recipe.
     *
     * @return the diagnostic name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the named dynamic dimensions in stable index order.
     *
     * @return the named dynamic dimensions
     */
    public List<Dimension> getDimensions() {
        return dimensions;
    }

    /**
     * Returns the invocation inputs in stable index order.
     *
     * @return the invocation inputs
     */
    public List<Input> getInputs() {
        return inputs;
    }

    /**
     * Returns the executable constants in stable index order.
     *
     * @return the executable constants
     */
    public List<Constant> getConstants() {
        return constants;
    }

    /**
     * Returns all values in topological and stable index order.
     *
     * <p>The list includes inputs, constants, and computed values.
     *
     * @return all recipe values
     */
    public List<Value> getValues() {
        return values;
    }

    /**
     * Returns the session outputs in stable index order.
     *
     * @return the session outputs
     */
    public List<Output> getOutputs() {
        return outputs;
    }

    private static <T> List<T> immutableCopy(List<T> values) {
        return Collections.unmodifiableList(new ArrayList<>(values));
    }

    /** A bounded leading dimension whose active extent is set for each invocation. */
    public static final class Dimension {

        private final Object owner;
        private final int index;
        private final String name;
        private final long maximumExtent;

        private Dimension(Object owner, int index, String name, long maximumExtent) {
            this.owner = owner;
            this.index = index;
            this.name = name;
            this.maximumExtent = maximumExtent;
        }

        /**
         * Returns the stable index of this dimension within its recipe.
         *
         * @return the stable index
         */
        public int getIndex() {
            return index;
        }

        /**
         * Returns the name of this dimension.
         *
         * @return the dimension name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the storage capacity of this dimension.
         *
         * @return the maximum extent
         */
        public long getMaximumExtent() {
            return maximumExtent;
        }
    }

    /** Describes the data type and bounded shape of a recipe value. */
    public static final class TensorSpec {

        private final DataType dataType;
        private final Dimension leadingDimension;
        private final long[] innerShape;

        private TensorSpec(DataType dataType, Dimension leadingDimension, long[] innerShape) {
            this.dataType = Objects.requireNonNull(dataType, "dataType");
            this.leadingDimension = leadingDimension;
            this.innerShape = validateShape(innerShape);
        }

        /**
         * Creates a tensor specification with one named leading dimension.
         *
         * @param dataType the element data type
         * @param leadingDimension the bounded leading dimension
         * @param innerShape the fixed dimensions following the leading dimension
         * @return a tensor specification
         */
        public static TensorSpec of(
                DataType dataType, Dimension leadingDimension, long... innerShape) {
            return new TensorSpec(
                    dataType,
                    Objects.requireNonNull(leadingDimension, "leadingDimension"),
                    innerShape);
        }

        /**
         * Creates a tensor specification with a fully fixed shape.
         *
         * @param dataType the element data type
         * @param shape the fixed shape
         * @return a tensor specification
         */
        public static TensorSpec fixed(DataType dataType, long... shape) {
            return new TensorSpec(dataType, null, shape);
        }

        /**
         * Returns the element data type.
         *
         * @return the element data type
         */
        public DataType getDataType() {
            return dataType;
        }

        /**
         * Returns the named leading dimension, or {@code null} for a fully fixed tensor.
         *
         * @return the leading dimension, or {@code null}
         */
        public Dimension getLeadingDimension() {
            return leadingDimension;
        }

        /**
         * Returns a copy of the fixed inner shape.
         *
         * @return the fixed inner shape
         */
        public long[] getInnerShape() {
            return innerShape.clone();
        }

        /**
         * Returns the maximum storage shape.
         *
         * @return the maximum storage shape
         */
        public Shape getMaximumShape() {
            if (leadingDimension == null) {
                return new Shape(innerShape.clone());
            }
            long[] shape = new long[innerShape.length + 1];
            shape[0] = leadingDimension.maximumExtent;
            System.arraycopy(innerShape, 0, shape, 1, innerShape.length);
            return new Shape(shape);
        }

        private static long[] validateShape(long[] shape) {
            Objects.requireNonNull(shape, "shape");
            long[] copy = shape.clone();
            for (long dimension : copy) {
                if (dimension <= 0) {
                    throw new IllegalArgumentException("Tensor dimensions must be positive.");
                }
            }
            return copy;
        }
    }

    /** A typed value in a fusion recipe. */
    public abstract static class Value {

        private final Object owner;
        private final int index;
        private final String name;
        private final TensorSpec spec;

        private Value(Object owner, int index, String name, TensorSpec spec) {
            this.owner = owner;
            this.index = index;
            this.name = name;
            this.spec = spec;
        }

        /**
         * Returns the stable index of this value within its recipe.
         *
         * @return the stable index
         */
        public int getIndex() {
            return index;
        }

        /**
         * Returns the diagnostic name of this value.
         *
         * @return the value name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the tensor specification of this value.
         *
         * @return the tensor specification
         */
        public TensorSpec getSpec() {
            return spec;
        }
    }

    /** A value supplied by each {@link FusionInvocation}. */
    public static final class Input extends Value {

        private final int inputIndex;

        private Input(Object owner, int index, int inputIndex, String name, TensorSpec spec) {
            super(owner, index, name, spec);
            this.inputIndex = inputIndex;
        }

        /**
         * Returns the stable index of this input within the input list.
         *
         * @return the stable input index
         */
        public int getInputIndex() {
            return inputIndex;
        }
    }

    /** A read-only value bound when a {@link FusionPlan} creates an executable. */
    public static final class Constant extends Value {

        private final int constantIndex;

        private Constant(Object owner, int index, int constantIndex, String name, TensorSpec spec) {
            super(owner, index, name, spec);
            this.constantIndex = constantIndex;
        }

        /**
         * Returns the stable index of this constant within the constant list.
         *
         * @return the stable constant index
         */
        public int getConstantIndex() {
            return constantIndex;
        }
    }

    /** An activation supported by affine fusion stages. */
    public enum Activation {
        /** Leaves the affine sum unchanged. */
        NONE,

        /** Applies the sigmoid linear unit activation. */
        SILU
    }

    /** One projected source in an {@link AffineSum} stage. */
    public static final class AffineTerm {

        private final Value input;
        private final Constant weight;

        private AffineTerm(Value input, Constant weight) {
            this.input = input;
            this.weight = weight;
        }

        /**
         * Returns the value projected by this term.
         *
         * @return the input value
         */
        public Value getInput() {
            return input;
        }

        /**
         * Returns the {@code [outputWidth, inputWidth]} projection weight.
         *
         * @return the projection weight
         */
        public Constant getWeight() {
            return weight;
        }
    }

    /**
     * A bounded affine projection sum with optional fixed-shape broadcasting and activation.
     *
     * <p>For terms {@code (X_i, W_i)}, this value computes {@code activation(bias + sum_i X_i
     * W_i^T)}. Dynamic inputs use the same named leading dimension. A fully fixed input may instead
     * use a singleton first axis, which is broadcast over the active leading extent. Including that
     * singleton axis, each fixed input has the same rank as the output. Dimensions between the
     * leading dimension and feature width follow standard fixed-shape broadcasting rules; no input
     * is expanded in the recipe. Source values may independently use FLOAT16, BFLOAT16, or FLOAT32.
     * All weights and the optional bias use one projection data type, inferred from the weights,
     * and the output uses that projection data type. A backend converts each source to the
     * projection data type before GEMM. The numerical contract permits rounding to that data type
     * at source-conversion, backend GEMM, or projection-group boundaries before the projected
     * values are added.
     */
    public static final class AffineSum extends Value {

        private final List<AffineTerm> terms;
        private final Constant bias;
        private final Activation activation;

        private AffineSum(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                List<AffineTerm> terms,
                Constant bias,
                Activation activation) {
            super(owner, index, name, spec);
            this.terms = immutableCopy(terms);
            this.bias = bias;
            this.activation = activation;
        }

        /**
         * Returns the projected terms in declaration order.
         *
         * @return the projected terms
         */
        public List<AffineTerm> getTerms() {
            return terms;
        }

        /**
         * Returns the optional one-dimensional bias.
         *
         * @return the bias, or {@code null}
         */
        public Constant getBias() {
            return bias;
        }

        /**
         * Returns the activation applied after summation.
         *
         * @return the activation
         */
        public Activation getActivation() {
            return activation;
        }
    }

    /** One row-gather source in an {@link IndexedAffine} stage. */
    public static final class IndexedAffineSource {

        private final Value input;
        private final long indexDivisor;

        private IndexedAffineSource(Value input, long indexDivisor) {
            this.input = input;
            this.indexDivisor = indexDivisor;
        }

        /**
         * Returns the two-dimensional floating-point source value.
         *
         * @return the source value
         */
        public Value getInput() {
            return input;
        }

        /**
         * Returns the divisor applied to each destination index before gathering this source.
         *
         * @return the positive index divisor
         */
        public long getIndexDivisor() {
            return indexDivisor;
        }
    }

    /**
     * A bounded indexed two-layer affine projection with a dense scatter result.
     *
     * <p>For each active row {@code p}, the stage reads a unique destination index {@code d} and
     * gathers row {@code floor(d / divisor_i)} from every source {@code X_i}. The gathered rows are
     * concatenated into {@code x}, and the stage computes {@code y = W_o activation(W_h x + b_h) +
     * b_o}. The row {@code y} is written to destination row {@code d}; unselected destination rows
     * are zero. Destination indices must be unique and within the active destination extent. Each
     * divided index must be within the active leading extent of its source.
     *
     * <p>Sources may independently use FLOAT16, BFLOAT16, or FLOAT32. Both weights, both optional
     * biases, and the result use one projection data type. A backend converts gathered sources to
     * that type before the hidden projection. Numerical results may round at source conversion,
     * GEMM, activation, or output-projection boundaries.
     */
    public static final class IndexedAffine extends Value {

        private final Value indices;
        private final List<IndexedAffineSource> sources;
        private final Constant hiddenWeight;
        private final Constant hiddenBias;
        private final Activation activation;
        private final Constant outputWeight;
        private final Constant outputBias;

        private IndexedAffine(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                Value indices,
                List<IndexedAffineSource> sources,
                Constant hiddenWeight,
                Constant hiddenBias,
                Activation activation,
                Constant outputWeight,
                Constant outputBias) {
            super(owner, index, name, spec);
            this.indices = indices;
            this.sources = immutableCopy(sources);
            this.hiddenWeight = hiddenWeight;
            this.hiddenBias = hiddenBias;
            this.activation = activation;
            this.outputWeight = outputWeight;
            this.outputBias = outputBias;
        }

        /**
         * Returns the one-dimensional INT32 or INT64 destination indices.
         *
         * @return the destination indices
         */
        public Value getIndices() {
            return indices;
        }

        /**
         * Returns the gathered sources in concatenation order.
         *
         * @return the gathered sources
         */
        public List<IndexedAffineSource> getSources() {
            return sources;
        }

        /**
         * Returns the {@code [hiddenWidth, concatenatedWidth]} hidden weight.
         *
         * @return the hidden weight
         */
        public Constant getHiddenWeight() {
            return hiddenWeight;
        }

        /**
         * Returns the optional {@code [hiddenWidth]} hidden bias.
         *
         * @return the hidden bias, or {@code null}
         */
        public Constant getHiddenBias() {
            return hiddenBias;
        }

        /**
         * Returns the activation applied to the hidden projection.
         *
         * @return the hidden activation
         */
        public Activation getActivation() {
            return activation;
        }

        /**
         * Returns the {@code [outputWidth, hiddenWidth]} output weight.
         *
         * @return the output weight
         */
        public Constant getOutputWeight() {
            return outputWeight;
        }

        /**
         * Returns the optional {@code [outputWidth]} output bias.
         *
         * @return the output bias, or {@code null}
         */
        public Constant getOutputBias() {
            return outputBias;
        }
    }

    /**
     * A value that packs two-dimensional floating-point sources into a contiguous FLOAT32 tensor.
     *
     * <p>All sources share the same named leading dimension. Their complete last axes are appended
     * in source order. Sources may independently use FLOAT16, BFLOAT16, or FLOAT32.
     */
    public static final class OutputPack extends Value {

        private final List<Value> sources;

        private OutputPack(
                Object owner, int index, String name, TensorSpec spec, List<Value> sources) {
            super(owner, index, name, spec);
            this.sources = immutableCopy(sources);
        }

        /**
         * Returns the source values in packing order.
         *
         * @return the source values
         */
        public List<Value> getSources() {
            return sources;
        }
    }

    /**
     * A presence-aware binary blend of two context values.
     *
     * <p>For baseline context {@code b}, selected context {@code s}, selected logit {@code l}, and
     * presence values {@code p_b} and {@code p_s}, this stage computes the FLOAT32 value
     *
     * <pre>
     * q   = sigmoid(l)
     * j   = p_b * p_s
     * w_b = p_b - j * q
     * w_s = p_s - j + j * q
     * y   = w_b * b + w_s * s
     * </pre>
     *
     * <p>The context values are {@code [batch, width]} tensors and may independently use FLOAT16,
     * BFLOAT16, or FLOAT32. The logit and presence values are {@code [batch, 1]} tensors that may
     * independently use those same data types; their last axis is broadcast over the context width.
     * When only one branch is present, that branch is copied without applying the logit. When
     * neither branch is present, the result is zero.
     */
    public static final class BinaryBranchBlend extends Value {

        private final Value baselineContext;
        private final Value selectedContext;
        private final Value selectedLogit;
        private final Value baselinePresence;
        private final Value selectedPresence;

        private BinaryBranchBlend(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                Value baselineContext,
                Value selectedContext,
                Value selectedLogit,
                Value baselinePresence,
                Value selectedPresence) {
            super(owner, index, name, spec);
            this.baselineContext = baselineContext;
            this.selectedContext = selectedContext;
            this.selectedLogit = selectedLogit;
            this.baselinePresence = baselinePresence;
            this.selectedPresence = selectedPresence;
        }

        /**
         * Returns the baseline context.
         *
         * @return the baseline context
         */
        public Value getBaselineContext() {
            return baselineContext;
        }

        /**
         * Returns the selected context.
         *
         * @return the selected context
         */
        public Value getSelectedContext() {
            return selectedContext;
        }

        /**
         * Returns the logit for selecting the selected context when both branches are present.
         *
         * @return the selected-branch logit
         */
        public Value getSelectedLogit() {
            return selectedLogit;
        }

        /**
         * Returns the baseline-branch presence value.
         *
         * @return the baseline presence value
         */
        public Value getBaselinePresence() {
            return baselinePresence;
        }

        /**
         * Returns the selected-branch presence value.
         *
         * @return the selected presence value
         */
        public Value getSelectedPresence() {
            return selectedPresence;
        }
    }

    /**
     * The immutable parameters of one readout in a {@link SingleQueryCrossAttentionReadoutGroup}.
     */
    public static final class SingleQueryCrossAttentionReadout {

        private final Constant querySeedWeight;
        private final Constant querySeedBias;
        private final Constant queryWeight;
        private final Constant queryBias;
        private final Constant keyValueWeight;
        private final Constant contextWeight;
        private final Constant contextBias;
        private final Constant queryNormWeight;
        private final Constant queryNormBias;
        private final Constant feedForwardNormWeight;
        private final Constant feedForwardNormBias;
        private final Constant feedForwardExpansionWeight;
        private final Constant feedForwardExpansionBias;
        private final Constant feedForwardProjectionWeight;
        private final Constant feedForwardProjectionBias;
        private final Constant outputNormWeight;
        private final Constant outputNormBias;
        private final int feedForwardWidth;

        private SingleQueryCrossAttentionReadout(
                Constant querySeedWeight,
                Constant querySeedBias,
                Constant queryWeight,
                Constant queryBias,
                Constant keyValueWeight,
                Constant contextWeight,
                Constant contextBias,
                Constant queryNormWeight,
                Constant queryNormBias,
                Constant feedForwardNormWeight,
                Constant feedForwardNormBias,
                Constant feedForwardExpansionWeight,
                Constant feedForwardExpansionBias,
                Constant feedForwardProjectionWeight,
                Constant feedForwardProjectionBias,
                Constant outputNormWeight,
                Constant outputNormBias,
                int feedForwardWidth) {
            this.querySeedWeight = querySeedWeight;
            this.querySeedBias = querySeedBias;
            this.queryWeight = queryWeight;
            this.queryBias = queryBias;
            this.keyValueWeight = keyValueWeight;
            this.contextWeight = contextWeight;
            this.contextBias = contextBias;
            this.queryNormWeight = queryNormWeight;
            this.queryNormBias = queryNormBias;
            this.feedForwardNormWeight = feedForwardNormWeight;
            this.feedForwardNormBias = feedForwardNormBias;
            this.feedForwardExpansionWeight = feedForwardExpansionWeight;
            this.feedForwardExpansionBias = feedForwardExpansionBias;
            this.feedForwardProjectionWeight = feedForwardProjectionWeight;
            this.feedForwardProjectionBias = feedForwardProjectionBias;
            this.outputNormWeight = outputNormWeight;
            this.outputNormBias = outputNormBias;
            this.feedForwardWidth = feedForwardWidth;
        }

        /**
         * Returns the {@code [hiddenWidth, 2 * hiddenWidth]} query-seed weight.
         *
         * @return the {@code [hiddenWidth, 2 * hiddenWidth]} query-seed weight
         */
        public Constant getQuerySeedWeight() {
            return querySeedWeight;
        }

        /**
         * Returns the query-seed bias.
         *
         * @return the query-seed bias
         */
        public Constant getQuerySeedBias() {
            return querySeedBias;
        }

        /**
         * Returns the {@code [attentionWidth, hiddenWidth]} query weight.
         *
         * @return the {@code [attentionWidth, hiddenWidth]} query weight
         */
        public Constant getQueryWeight() {
            return queryWeight;
        }

        /**
         * Returns the query bias.
         *
         * @return the query bias
         */
        public Constant getQueryBias() {
            return queryBias;
        }

        /**
         * Returns the {@code [2 * attentionWidth, hiddenWidth]} key-value weight.
         *
         * @return the {@code [2 * attentionWidth, hiddenWidth]} key-value weight
         */
        public Constant getKeyValueWeight() {
            return keyValueWeight;
        }

        /**
         * Returns the {@code [hiddenWidth, attentionWidth]} context weight.
         *
         * @return the {@code [hiddenWidth, attentionWidth]} context weight
         */
        public Constant getContextWeight() {
            return contextWeight;
        }

        /**
         * Returns the context bias.
         *
         * @return the context bias
         */
        public Constant getContextBias() {
            return contextBias;
        }

        /**
         * Returns the post-attention LayerNorm scale.
         *
         * @return the post-attention LayerNorm scale
         */
        public Constant getQueryNormWeight() {
            return queryNormWeight;
        }

        /**
         * Returns the post-attention LayerNorm bias.
         *
         * @return the post-attention LayerNorm bias
         */
        public Constant getQueryNormBias() {
            return queryNormBias;
        }

        /**
         * Returns the feed-forward-input LayerNorm scale.
         *
         * @return the feed-forward-input LayerNorm scale
         */
        public Constant getFeedForwardNormWeight() {
            return feedForwardNormWeight;
        }

        /**
         * Returns the feed-forward-input LayerNorm bias.
         *
         * @return the feed-forward-input LayerNorm bias
         */
        public Constant getFeedForwardNormBias() {
            return feedForwardNormBias;
        }

        /**
         * Returns the {@code [feedForwardWidth, hiddenWidth]} expansion weight.
         *
         * @return the {@code [feedForwardWidth, hiddenWidth]} expansion weight
         */
        public Constant getFeedForwardExpansionWeight() {
            return feedForwardExpansionWeight;
        }

        /**
         * Returns the feed-forward expansion bias.
         *
         * @return the feed-forward expansion bias
         */
        public Constant getFeedForwardExpansionBias() {
            return feedForwardExpansionBias;
        }

        /**
         * Returns the {@code [hiddenWidth, feedForwardWidth]} projection weight.
         *
         * @return the {@code [hiddenWidth, feedForwardWidth]} projection weight
         */
        public Constant getFeedForwardProjectionWeight() {
            return feedForwardProjectionWeight;
        }

        /**
         * Returns the feed-forward projection bias.
         *
         * @return the feed-forward projection bias
         */
        public Constant getFeedForwardProjectionBias() {
            return feedForwardProjectionBias;
        }

        /**
         * Returns the output LayerNorm scale.
         *
         * @return the output LayerNorm scale
         */
        public Constant getOutputNormWeight() {
            return outputNormWeight;
        }

        /**
         * Returns the output LayerNorm bias.
         *
         * @return the output LayerNorm bias
         */
        public Constant getOutputNormBias() {
            return outputNormBias;
        }

        /**
         * Returns the feed-forward hidden width.
         *
         * @return the feed-forward hidden width
         */
        public int getFeedForwardWidth() {
            return feedForwardWidth;
        }
    }

    /**
     * Several purpose-specific single-query cross-attention readouts over one shared memory.
     *
     * <p>The stage first computes the masked mean of {@code memory} once. Each readout projects the
     * concatenation of the selected {@code querySource} token and that mean, attends to the same
     * memory, applies a residual LayerNorm, evaluates a SiLU feed-forward network, and applies a
     * final residual LayerNorm. The memory and query source use one floating-point input type. A
     * nonzero mask element marks a valid memory token. A row with no valid tokens uses a zero mean
     * and a zero attention context.
     *
     * <p>For each attention head, an implementation may use {@code q (E W_k^T)^T = E (W_k^T q)} and
     * {@code softmax(E (W_k^T q)) E W_v^T = (softmax(E (W_k^T q)) E) W_v^T}. This avoids the
     * projected key-value tensor while preserving exact real-number semantics. The memory and query
     * source may use a data type different from the projections. They are converted to the
     * projection data type at the same boundaries as an autocast linear operation; the masked mean
     * is reduced in FLOAT32 before that conversion. FLOAT16 and BFLOAT16 implementations accumulate
     * projections, reductions, and LayerNorm statistics in FLOAT32 and may round at the documented
     * projection and residual boundaries; they are not required to be bitwise identical to a
     * materialized key-value implementation.
     *
     * <p>Each readout produces a separate contiguous {@code [batch, hiddenWidth]} state. Keeping
     * the states separate allows each state to be passed directly to a later Fusion recipe without
     * materializing a slice. A backend may keep the states in one readout-major backing allocation
     * and expose contiguous aliases. Readouts may use different feed-forward widths; a backend may
     * pad them to the largest declared width to execute the group without per-readout dispatches.
     */
    public static final class SingleQueryCrossAttentionReadoutGroup {

        private final Value memory;
        private final Value querySource;
        private final Value validMask;
        private final int queryIndex;
        private final int attentionHeads;
        private final int attentionWidth;
        private final int maximumFeedForwardWidth;
        private final float epsilon;
        private final List<SingleQueryCrossAttentionReadout> readouts;
        private final List<SingleQueryCrossAttentionReadoutState> states;

        private SingleQueryCrossAttentionReadoutGroup(
                Object owner,
                int firstValueIndex,
                List<String> stateNames,
                TensorSpec stateSpec,
                Value memory,
                Value querySource,
                Value validMask,
                int queryIndex,
                int attentionHeads,
                int attentionWidth,
                int maximumFeedForwardWidth,
                float epsilon,
                List<SingleQueryCrossAttentionReadout> readouts) {
            this.memory = memory;
            this.querySource = querySource;
            this.validMask = validMask;
            this.queryIndex = queryIndex;
            this.attentionHeads = attentionHeads;
            this.attentionWidth = attentionWidth;
            this.maximumFeedForwardWidth = maximumFeedForwardWidth;
            this.epsilon = epsilon;
            this.readouts = immutableCopy(readouts);
            List<SingleQueryCrossAttentionReadoutState> stateValues =
                    new ArrayList<>(readouts.size());
            for (int index = 0; index < readouts.size(); ++index) {
                stateValues.add(
                        new SingleQueryCrossAttentionReadoutState(
                                owner,
                                firstValueIndex + index,
                                stateNames.get(index),
                                stateSpec,
                                this,
                                index));
            }
            states = immutableCopy(stateValues);
        }

        /**
         * Returns the shared {@code [batch, tokens, hiddenWidth]} memory.
         *
         * @return the shared {@code [batch, tokens, hiddenWidth]} memory
         */
        public Value getMemory() {
            return memory;
        }

        /**
         * Returns the shared {@code [batch, hiddenWidth]} or {@code [batch, queryTokens,
         * hiddenWidth]} query source.
         *
         * @return the shared {@code [batch, hiddenWidth]} or {@code [batch, queryTokens,
         *     hiddenWidth]} query source
         */
        public Value getQuerySource() {
            return querySource;
        }

        /**
         * Returns the shared nonzero-is-valid {@code [batch, tokens]} mask.
         *
         * @return the shared nonzero-is-valid {@code [batch, tokens]} mask
         */
        public Value getValidMask() {
            return validMask;
        }

        /**
         * Returns the selected query token index, or zero for a two-dimensional query source.
         *
         * @return the selected query token index, or zero for a two-dimensional query source
         */
        public int getQueryIndex() {
            return queryIndex;
        }

        /**
         * Returns the attention head count.
         *
         * @return the attention head count
         */
        public int getAttentionHeads() {
            return attentionHeads;
        }

        /**
         * Returns the concatenated attention width.
         *
         * @return the concatenated attention width
         */
        public int getAttentionWidth() {
            return attentionWidth;
        }

        /**
         * Returns the largest feed-forward width in the group.
         *
         * @return the largest feed-forward width in the group
         */
        public int getMaximumFeedForwardWidth() {
            return maximumFeedForwardWidth;
        }

        /**
         * Returns the LayerNorm epsilon.
         *
         * @return the LayerNorm epsilon
         */
        public float getEpsilon() {
            return epsilon;
        }

        /**
         * Returns the readouts in output-axis order.
         *
         * @return the readouts in output-axis order
         */
        public List<SingleQueryCrossAttentionReadout> getReadouts() {
            return readouts;
        }

        /**
         * Returns the contiguous state produced by one readout.
         *
         * @param index the readout index
         * @return the {@code [batch, hiddenWidth]} state
         */
        public SingleQueryCrossAttentionReadoutState getReadoutState(int index) {
            return states.get(index);
        }

        /**
         * Returns the contiguous readout states in readout order.
         *
         * @return the contiguous readout states in readout order
         */
        public List<SingleQueryCrossAttentionReadoutState> getReadoutStates() {
            return states;
        }
    }

    /** One contiguous state produced by a {@link SingleQueryCrossAttentionReadoutGroup}. */
    public static final class SingleQueryCrossAttentionReadoutState extends Value {

        private final SingleQueryCrossAttentionReadoutGroup group;
        private final int readoutIndex;

        private SingleQueryCrossAttentionReadoutState(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                SingleQueryCrossAttentionReadoutGroup group,
                int readoutIndex) {
            super(owner, index, name, spec);
            this.group = group;
            this.readoutIndex = readoutIndex;
        }

        /**
         * Returns the group that computes this state.
         *
         * @return the group that computes this state
         */
        public SingleQueryCrossAttentionReadoutGroup getGroup() {
            return group;
        }

        /**
         * Returns this state's index within the group.
         *
         * @return this state's index within the group
         */
        public int getReadoutIndex() {
            return readoutIndex;
        }
    }

    /**
     * One indexed transformer encoder block applied independently to fixed-size local groups.
     *
     * <p>The input has shape {@code [batch, groups, tokens, hiddenWidth]}. The one-dimensional
     * integer index contains the unique, ascending, zero-based rows selected from the flattened
     * {@code [batch, groups, tokens]} prefix. Each selected token is first normalized by the input
     * LayerNorm. The encoder then applies one pre-normalized self-attention block and one SiLU
     * feed-forward block within its local group. Unselected output tokens are zero and never
     * participate as attention keys or queries.
     *
     * <p>The index uses an independent bounded leading dimension so an invocation can bind a view
     * over a larger packed index slab without copying it. This stage preserves the real-number
     * semantics of gathering selected rows, encoding compact rows, and scattering the result back
     * into the dense layout. FLOAT16 and BFLOAT16 implementations accumulate reductions and
     * LayerNorm statistics in FLOAT32 and may round at projection and residual boundaries.
     */
    public static final class IndexedLocalTransformerEncoder extends Value {

        private final List<Value> inputSegments;
        private final Value indices;
        private final Constant inputNormWeight;
        private final Constant inputNormBias;
        private final TransformerEncoderBlock block;
        private final int attentionHeads;
        private final int attentionWidth;
        private final int feedForwardWidth;
        private final float epsilon;

        private IndexedLocalTransformerEncoder(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                List<Value> inputSegments,
                Value indices,
                Constant inputNormWeight,
                Constant inputNormBias,
                TransformerEncoderBlock block,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth,
                float epsilon) {
            super(owner, index, name, spec);
            this.inputSegments = immutableCopy(inputSegments);
            this.indices = indices;
            this.inputNormWeight = inputNormWeight;
            this.inputNormBias = inputNormBias;
            this.block = block;
            this.attentionHeads = attentionHeads;
            this.attentionWidth = attentionWidth;
            this.feedForwardWidth = feedForwardWidth;
            this.epsilon = epsilon;
        }

        /**
         * Returns the only dense local-group input.
         *
         * @return the dense local-group input
         * @throws IllegalStateException if this encoder uses more than one input segment
         */
        public Value getInput() {
            if (inputSegments.size() != 1) {
                throw new IllegalStateException(
                        "The indexed local transformer input is segmented.");
            }
            return inputSegments.get(0);
        }

        /**
         * Returns the dense local-group input segments in token order.
         *
         * <p>Each segment has shape {@code [batch, groups, segmentTokens, hiddenWidth]}. The
         * concatenation is logical: engines can gather indexed rows directly from the segments
         * without materializing one dense input.
         *
         * @return the immutable input-segment list
         */
        public List<Value> getInputSegments() {
            return inputSegments;
        }

        /**
         * @return the ascending flattened token indices
         */
        public Value getIndices() {
            return indices;
        }

        /**
         * @return the input LayerNorm scale
         */
        public Constant getInputNormWeight() {
            return inputNormWeight;
        }

        /**
         * @return the input LayerNorm bias
         */
        public Constant getInputNormBias() {
            return inputNormBias;
        }

        /**
         * @return the transformer block parameters
         */
        public TransformerEncoderBlock getBlock() {
            return block;
        }

        /**
         * @return the attention head count
         */
        public int getAttentionHeads() {
            return attentionHeads;
        }

        /**
         * @return the concatenated attention width
         */
        public int getAttentionWidth() {
            return attentionWidth;
        }

        /**
         * @return the feed-forward hidden width
         */
        public int getFeedForwardWidth() {
            return feedForwardWidth;
        }

        /**
         * @return the LayerNorm epsilon
         */
        public float getEpsilon() {
            return epsilon;
        }
    }

    /** The immutable parameters of one block in a {@link TransformerEncoderStack}. */
    public static final class TransformerEncoderBlock {

        private final Constant attentionInputWeight;
        private final Constant attentionInputBias;
        private final Constant queryKeyValueWeight;
        private final Constant attentionOutputWeight;
        private final Constant attentionOutputBias;
        private final Constant feedForwardInputWeight;
        private final Constant feedForwardInputBias;
        private final Constant feedForwardExpansionWeight;
        private final Constant feedForwardExpansionBias;
        private final Constant feedForwardProjectionWeight;
        private final Constant feedForwardProjectionBias;
        private final Constant outputWeight;
        private final Constant outputBias;

        private TransformerEncoderBlock(
                Constant attentionInputWeight,
                Constant attentionInputBias,
                Constant queryKeyValueWeight,
                Constant attentionOutputWeight,
                Constant attentionOutputBias,
                Constant feedForwardInputWeight,
                Constant feedForwardInputBias,
                Constant feedForwardExpansionWeight,
                Constant feedForwardExpansionBias,
                Constant feedForwardProjectionWeight,
                Constant feedForwardProjectionBias,
                Constant outputWeight,
                Constant outputBias) {
            this.attentionInputWeight = attentionInputWeight;
            this.attentionInputBias = attentionInputBias;
            this.queryKeyValueWeight = queryKeyValueWeight;
            this.attentionOutputWeight = attentionOutputWeight;
            this.attentionOutputBias = attentionOutputBias;
            this.feedForwardInputWeight = feedForwardInputWeight;
            this.feedForwardInputBias = feedForwardInputBias;
            this.feedForwardExpansionWeight = feedForwardExpansionWeight;
            this.feedForwardExpansionBias = feedForwardExpansionBias;
            this.feedForwardProjectionWeight = feedForwardProjectionWeight;
            this.feedForwardProjectionBias = feedForwardProjectionBias;
            this.outputWeight = outputWeight;
            this.outputBias = outputBias;
        }

        /**
         * @return the affine scale of the attention-input LayerNorm
         */
        public Constant getAttentionInputWeight() {
            return attentionInputWeight;
        }

        /**
         * @return the affine bias of the attention-input LayerNorm
         */
        public Constant getAttentionInputBias() {
            return attentionInputBias;
        }

        /**
         * @return the {@code [3 * attentionWidth, hiddenWidth]} QKV weight
         */
        public Constant getQueryKeyValueWeight() {
            return queryKeyValueWeight;
        }

        /**
         * @return the {@code [hiddenWidth, attentionWidth]} attention output weight
         */
        public Constant getAttentionOutputWeight() {
            return attentionOutputWeight;
        }

        /**
         * @return the attention output bias
         */
        public Constant getAttentionOutputBias() {
            return attentionOutputBias;
        }

        /**
         * @return the affine scale of the feed-forward-input LayerNorm
         */
        public Constant getFeedForwardInputWeight() {
            return feedForwardInputWeight;
        }

        /**
         * @return the affine bias of the feed-forward-input LayerNorm
         */
        public Constant getFeedForwardInputBias() {
            return feedForwardInputBias;
        }

        /**
         * @return the {@code [feedForwardWidth, hiddenWidth]} expansion weight
         */
        public Constant getFeedForwardExpansionWeight() {
            return feedForwardExpansionWeight;
        }

        /**
         * @return the feed-forward expansion bias
         */
        public Constant getFeedForwardExpansionBias() {
            return feedForwardExpansionBias;
        }

        /**
         * @return the {@code [hiddenWidth, feedForwardWidth]} projection weight
         */
        public Constant getFeedForwardProjectionWeight() {
            return feedForwardProjectionWeight;
        }

        /**
         * @return the feed-forward projection bias
         */
        public Constant getFeedForwardProjectionBias() {
            return feedForwardProjectionBias;
        }

        /**
         * @return the affine scale of the output LayerNorm
         */
        public Constant getOutputWeight() {
            return outputWeight;
        }

        /**
         * @return the affine bias of the output LayerNorm
         */
        public Constant getOutputBias() {
            return outputBias;
        }
    }

    /**
     * A stack of pre-normalized transformer encoder blocks over one fixed-length dense sequence.
     *
     * <p>Each block computes attention from {@code LayerNorm(x)}, adds its projected result to
     * {@code x}, normalizes that residual for a SiLU feed-forward network, then adds and normalizes
     * the projected feed-forward result. Projection weights use the input data type. LayerNorm
     * affine parameters may either use that data type or FLOAT32. The leading batch dimension is
     * bounded while token count and every feature width are fixed by the recipe.
     */
    public static final class TransformerEncoderStack extends Value {

        private final Value input;
        private final int attentionHeads;
        private final int attentionWidth;
        private final int feedForwardWidth;
        private final float epsilon;
        private final List<TransformerEncoderBlock> blocks;

        private TransformerEncoderStack(
                Object owner,
                int index,
                String name,
                TensorSpec spec,
                Value input,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth,
                float epsilon,
                List<TransformerEncoderBlock> blocks) {
            super(owner, index, name, spec);
            this.input = input;
            this.attentionHeads = attentionHeads;
            this.attentionWidth = attentionWidth;
            this.feedForwardWidth = feedForwardWidth;
            this.epsilon = epsilon;
            this.blocks = immutableCopy(blocks);
        }

        /**
         * @return the stack input
         */
        public Value getInput() {
            return input;
        }

        /**
         * @return the attention head count
         */
        public int getAttentionHeads() {
            return attentionHeads;
        }

        /**
         * @return the concatenated attention feature width
         */
        public int getAttentionWidth() {
            return attentionWidth;
        }

        /**
         * @return the feed-forward hidden width
         */
        public int getFeedForwardWidth() {
            return feedForwardWidth;
        }

        /**
         * @return the LayerNorm epsilon
         */
        public float getEpsilon() {
            return epsilon;
        }

        /**
         * @return the blocks in execution order
         */
        public List<TransformerEncoderBlock> getBlocks() {
            return blocks;
        }
    }

    /** A named session output backed by slot-owned persistent storage. */
    public static final class Output {

        private final Object owner;
        private final int index;
        private final String name;
        private final Value value;

        private Output(Object owner, int index, String name, Value value) {
            this.owner = owner;
            this.index = index;
            this.name = name;
            this.value = value;
        }

        /**
         * Returns the stable index of this output within its recipe.
         *
         * @return the stable index
         */
        public int getIndex() {
            return index;
        }

        /**
         * Returns the output name.
         *
         * @return the output name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the value materialized into this output.
         *
         * @return the output value
         */
        public Value getValue() {
            return value;
        }

        /**
         * Returns the tensor specification of this output.
         *
         * @return the output tensor specification
         */
        public TensorSpec getSpec() {
            return value.spec;
        }
    }

    /** Builds an immutable {@link FusionRecipe}. */
    public static final class Builder {

        private final Object owner;
        private final String name;
        private final Set<String> valueNames;
        private final Set<String> dimensionNames;
        private final Set<String> outputNames;
        private final List<Dimension> dimensions;
        private final List<Input> inputs;
        private final List<Constant> constants;
        private final List<Value> values;
        private final List<Output> outputs;
        private boolean built;

        private Builder(String name) {
            this.name = requireName(name, "recipe");
            owner = new Object();
            valueNames = new HashSet<>();
            dimensionNames = new HashSet<>();
            outputNames = new HashSet<>();
            dimensions = new ArrayList<>();
            inputs = new ArrayList<>();
            constants = new ArrayList<>();
            values = new ArrayList<>();
            outputs = new ArrayList<>();
        }

        /**
         * Adds a bounded leading dimension.
         *
         * @param name the dimension name
         * @param maximumExtent the storage capacity of the dimension
         * @return the dimension handle
         */
        public Dimension addDimension(String name, long maximumExtent) {
            checkMutable();
            String checkedName = requireName(name, "dimension");
            if (maximumExtent <= 0) {
                throw new IllegalArgumentException("The maximum extent must be positive.");
            }
            if (!dimensionNames.add(checkedName)) {
                throw new IllegalArgumentException("Duplicate dimension name: " + checkedName);
            }
            Dimension dimension =
                    new Dimension(owner, dimensions.size(), checkedName, maximumExtent);
            dimensions.add(dimension);
            return dimension;
        }

        /**
         * Adds a value supplied by each invocation.
         *
         * @param name the input name
         * @param spec the input tensor specification
         * @return the input handle
         */
        public Input addInput(String name, TensorSpec spec) {
            checkMutable();
            checkSpec(spec);
            String checkedName = addValueName(name);
            Input input = new Input(owner, values.size(), inputs.size(), checkedName, spec);
            inputs.add(input);
            values.add(input);
            return input;
        }

        /**
         * Adds a read-only tensor bound when a plan creates an executable.
         *
         * @param name the constant name
         * @param spec the constant tensor specification
         * @return the constant handle
         */
        public Constant addConstant(String name, TensorSpec spec) {
            checkMutable();
            checkSpec(spec);
            String checkedName = addValueName(name);
            Constant constant =
                    new Constant(owner, values.size(), constants.size(), checkedName, spec);
            constants.add(constant);
            values.add(constant);
            return constant;
        }

        /**
         * Starts an affine-sum stage.
         *
         * <p>The returned builder accepts one or more source and weight pairs. Validation and value
         * insertion complete when {@link AffineSumBuilder#build()} is called.
         *
         * @param name the value name
         * @param outputWidth the common projection width
         * @return a builder for the affine-sum value
         */
        public AffineSumBuilder affineSum(String name, long outputWidth) {
            checkMutable();
            if (outputWidth <= 0) {
                throw new IllegalArgumentException("The affine output width must be positive.");
            }
            return new AffineSumBuilder(this, requireName(name, "value"), outputWidth);
        }

        /**
         * Starts an indexed-affine stage.
         *
         * <p>The indices use their own bounded leading dimension. The result uses {@code
         * destinationDimension} and has the output width declared by the output projection. The
         * returned builder validates and inserts the value when {@link
         * IndexedAffineBuilder#build()} is called.
         *
         * @param name the value name
         * @param indices one-dimensional INT32 or INT64 destination indices
         * @param destinationDimension the bounded leading dimension of the dense result
         * @return a builder for the indexed-affine value
         */
        public IndexedAffineBuilder indexedAffine(
                String name, Value indices, Dimension destinationDimension) {
            checkMutable();
            checkValue(indices);
            Objects.requireNonNull(destinationDimension, "destinationDimension");
            if (destinationDimension.owner != owner) {
                throw new IllegalArgumentException(
                        "The destination dimension belongs to a different recipe builder.");
            }
            return new IndexedAffineBuilder(
                    this, requireName(name, "value"), indices, destinationDimension);
        }

        /**
         * Starts a group of single-query cross-attention readouts.
         *
         * @param name the value name
         * @param memory a bounded {@code [batch, tokens, hiddenWidth]} floating-point value
         * @param querySource a bounded {@code [batch, hiddenWidth]} or {@code [batch, queryTokens,
         *     hiddenWidth]} floating-point value
         * @param validMask a bounded {@code [batch, tokens]} value whose nonzero elements are valid
         * @param attentionHeads the attention head count
         * @return a builder for the readout group
         */
        public SingleQueryCrossAttentionReadoutGroupBuilder singleQueryCrossAttentionReadoutGroup(
                String name, Value memory, Value querySource, Value validMask, int attentionHeads) {
            checkMutable();
            checkValue(memory);
            checkValue(querySource);
            checkValue(validMask);
            return new SingleQueryCrossAttentionReadoutGroupBuilder(
                    this,
                    requireName(name, "value"),
                    memory,
                    querySource,
                    validMask,
                    attentionHeads);
        }

        /**
         * Starts an indexed transformer encoder over fixed-size local groups.
         *
         * @param name the value name
         * @param input a bounded {@code [batch, groups, tokens, hiddenWidth]} floating-point value
         * @param indices bounded, ascending, flattened INT32 or INT64 token indices
         * @param attentionHeads the attention head count
         * @param attentionWidth the concatenated Q, K, and V feature width
         * @param feedForwardWidth the SiLU feed-forward hidden width
         * @return a builder for the indexed local transformer encoder
         */
        public IndexedLocalTransformerEncoderBuilder indexedLocalTransformerEncoder(
                String name,
                Value input,
                Value indices,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth) {
            checkMutable();
            checkValue(input);
            checkValue(indices);
            return new IndexedLocalTransformerEncoderBuilder(
                    this,
                    requireName(name, "value"),
                    Collections.singletonList(input),
                    indices,
                    attentionHeads,
                    attentionWidth,
                    feedForwardWidth);
        }

        /**
         * Starts an indexed transformer encoder over logically concatenated local-group segments.
         *
         * <p>Every segment must have shape {@code [batch, groups, segmentTokens, hiddenWidth]} with
         * the same bounded batch dimension, group count, hidden width, and data type. Flattened
         * indices address the logical concatenation in segment declaration order.
         *
         * @param name the value name
         * @param inputSegments bounded floating-point values in logical token order
         * @param indices bounded, ascending, flattened INT32 or INT64 token indices
         * @param attentionHeads the attention head count
         * @param attentionWidth the concatenated Q, K, and V feature width
         * @param feedForwardWidth the SiLU feed-forward hidden width
         * @return a builder for the indexed local transformer encoder
         */
        public IndexedLocalTransformerEncoderBuilder indexedLocalTransformerEncoder(
                String name,
                List<? extends Value> inputSegments,
                Value indices,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth) {
            checkMutable();
            Objects.requireNonNull(inputSegments, "inputSegments");
            if (inputSegments.isEmpty()) {
                throw new IllegalArgumentException(
                        "Indexed local transformer input segments must not be empty.");
            }
            List<Value> checkedSegments = new ArrayList<>(inputSegments.size());
            for (Value inputSegment : inputSegments) {
                checkValue(inputSegment);
                checkedSegments.add(inputSegment);
            }
            checkValue(indices);
            return new IndexedLocalTransformerEncoderBuilder(
                    this,
                    requireName(name, "value"),
                    checkedSegments,
                    indices,
                    attentionHeads,
                    attentionWidth,
                    feedForwardWidth);
        }

        /**
         * Starts a fixed-sequence transformer encoder stack.
         *
         * @param name the value name
         * @param input a bounded {@code [batch, tokens, hiddenWidth]} floating-point value
         * @param attentionHeads the attention head count
         * @param attentionWidth the concatenated Q, K, and V feature width
         * @param feedForwardWidth the SiLU feed-forward hidden width
         * @return a builder for the transformer encoder stack
         */
        public TransformerEncoderStackBuilder transformerEncoderStack(
                String name,
                Value input,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth) {
            checkMutable();
            checkValue(input);
            return new TransformerEncoderStackBuilder(
                    this,
                    requireName(name, "value"),
                    input,
                    attentionHeads,
                    attentionWidth,
                    feedForwardWidth);
        }

        /**
         * Adds a presence-aware binary branch blend.
         *
         * <p>The contexts must be two-dimensional floating-point values with the same leading
         * dimension and width. The selected logit and both presence values must be floating-point
         * {@code [batch, 1]} values using that leading dimension. They may use different data
         * types. The resulting value is FLOAT32 {@code [batch, width]}.
         *
         * @param name the value name
         * @param baselineContext the baseline {@code [batch, width]} context
         * @param selectedContext the selected {@code [batch, width]} context
         * @param selectedLogit the selected-branch {@code [batch, 1]} logit
         * @param baselinePresence the baseline {@code [batch, 1]} presence value
         * @param selectedPresence the selected {@code [batch, 1]} presence value
         * @return the blended FLOAT32 value
         */
        public BinaryBranchBlend binaryBranchBlend(
                String name,
                Value baselineContext,
                Value selectedContext,
                Value selectedLogit,
                Value baselinePresence,
                Value selectedPresence) {
            checkMutable();
            checkValue(baselineContext);
            checkValue(selectedContext);
            checkValue(selectedLogit);
            checkValue(baselinePresence);
            checkValue(selectedPresence);

            TensorSpec baselineSpec = baselineContext.spec;
            TensorSpec selectedSpec = selectedContext.spec;
            if (!isFloatingDataType(baselineSpec.dataType)
                    || !isFloatingDataType(selectedSpec.dataType)
                    || baselineSpec.leadingDimension == null
                    || baselineSpec.innerShape.length != 1
                    || selectedSpec.leadingDimension != baselineSpec.leadingDimension
                    || selectedSpec.innerShape.length != 1
                    || selectedSpec.innerShape[0] != baselineSpec.innerShape[0]) {
                throw new IllegalArgumentException(
                        "Binary branch contexts must be matching two-dimensional floating-point"
                                + " values.");
            }
            checkBinaryScalar(selectedLogit, baselineSpec.leadingDimension, "selected logit");
            checkBinaryScalar(baselinePresence, baselineSpec.leadingDimension, "baseline presence");
            checkBinaryScalar(selectedPresence, baselineSpec.leadingDimension, "selected presence");

            String checkedName = addValueName(name);
            BinaryBranchBlend value =
                    new BinaryBranchBlend(
                            owner,
                            values.size(),
                            checkedName,
                            TensorSpec.of(
                                    DataType.FLOAT32,
                                    baselineSpec.leadingDimension,
                                    baselineSpec.innerShape[0]),
                            baselineContext,
                            selectedContext,
                            selectedLogit,
                            baselinePresence,
                            selectedPresence);
            values.add(value);
            return value;
        }

        /**
         * Adds an output-pack value.
         *
         * <p>Each source must be a two-dimensional FLOAT16, BFLOAT16, or FLOAT32 value from this
         * builder. Sources must share the same named leading dimension. The resulting value is a
         * contiguous FLOAT32 tensor whose last-axis width is the sum of the source widths.
         *
         * @param name the value name
         * @param sources the values to pack in order
         * @return the packed value
         */
        public OutputPack outputPack(String name, Value... sources) {
            checkMutable();
            Objects.requireNonNull(sources, "sources");
            if (sources.length == 0) {
                throw new IllegalArgumentException("Output pack requires at least one source.");
            }

            List<Value> checkedSources = new ArrayList<>(sources.length);
            Dimension leadingDimension = null;
            long width = 0;
            for (Value source : sources) {
                checkValue(source);
                TensorSpec spec = source.spec;
                if (spec.leadingDimension == null || spec.innerShape.length != 1) {
                    throw new IllegalArgumentException(
                            "Output pack sources must have one leading and one inner dimension.");
                }
                if (!isFloatingDataType(spec.dataType)) {
                    throw new IllegalArgumentException(
                            "Output pack only supports FLOAT16, BFLOAT16, and FLOAT32 sources.");
                }
                if (leadingDimension == null) {
                    leadingDimension = spec.leadingDimension;
                } else if (leadingDimension != spec.leadingDimension) {
                    throw new IllegalArgumentException(
                            "Output pack sources must share the same leading dimension.");
                }
                width = Math.addExact(width, spec.innerShape[0]);
                checkedSources.add(source);
            }

            String checkedName = addValueName(name);
            TensorSpec outputSpec = TensorSpec.of(DataType.FLOAT32, leadingDimension, width);
            OutputPack value =
                    new OutputPack(owner, values.size(), checkedName, outputSpec, checkedSources);
            values.add(value);
            return value;
        }

        /**
         * Exposes a value as a session output.
         *
         * @param name the output name
         * @param value the value to materialize
         * @return the output handle
         */
        public Output addOutput(String name, Value value) {
            checkMutable();
            checkValue(value);
            String checkedName = requireName(name, "output");
            if (!outputNames.add(checkedName)) {
                throw new IllegalArgumentException("Duplicate output name: " + checkedName);
            }
            Output output = new Output(owner, outputs.size(), checkedName, value);
            outputs.add(output);
            return output;
        }

        /**
         * Builds the immutable recipe.
         *
         * @return the fusion recipe
         */
        public FusionRecipe build() {
            checkMutable();
            if (outputs.isEmpty()) {
                throw new IllegalStateException("A fusion recipe must have at least one output.");
            }
            built = true;
            return new FusionRecipe(this);
        }

        private void checkSpec(TensorSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (spec.leadingDimension != null && spec.leadingDimension.owner != owner) {
                throw new IllegalArgumentException(
                        "The tensor dimension belongs to a different recipe builder.");
            }
        }

        private void checkValue(Value value) {
            Objects.requireNonNull(value, "value");
            if (value.owner != owner) {
                throw new IllegalArgumentException(
                        "The value belongs to a different recipe builder.");
            }
        }

        private static void checkBinaryScalar(
                Value value, Dimension leadingDimension, String kind) {
            TensorSpec spec = value.spec;
            if (!isFloatingDataType(spec.dataType)
                    || spec.leadingDimension != leadingDimension
                    || spec.innerShape.length != 1
                    || spec.innerShape[0] != 1) {
                throw new IllegalArgumentException(
                        "Binary branch "
                                + kind
                                + " must be a floating-point [batch, 1] value using the context"
                                + " dimension.");
            }
        }

        private String addValueName(String name) {
            String checkedName = requireName(name, "value");
            if (!valueNames.add(checkedName)) {
                throw new IllegalArgumentException("Duplicate value name: " + checkedName);
            }
            return checkedName;
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException("The fusion recipe has already been built.");
            }
        }

        private static boolean isFloatingDataType(DataType dataType) {
            return dataType == DataType.FLOAT16
                    || dataType == DataType.BFLOAT16
                    || dataType == DataType.FLOAT32;
        }

        private static boolean isAffineDataType(DataType dataType) {
            return dataType == DataType.FLOAT16
                    || dataType == DataType.BFLOAT16
                    || dataType == DataType.FLOAT32;
        }

        private static boolean isMaskDataType(DataType dataType) {
            return dataType == DataType.BOOLEAN
                    || dataType == DataType.UINT8
                    || isAffineDataType(dataType);
        }

        private static String requireName(String name, String kind) {
            Objects.requireNonNull(name, kind + " name");
            if (name.trim().isEmpty()) {
                throw new IllegalArgumentException("The " + kind + " name must not be empty.");
            }
            return name;
        }
    }

    /** Builds one {@link SingleQueryCrossAttentionReadoutGroup} value within a {@link Builder}. */
    public static final class SingleQueryCrossAttentionReadoutGroupBuilder {

        private static final float DEFAULT_EPSILON = 1.0e-5f;
        private static final int MAXIMUM_READOUTS = 8;

        private final Builder recipeBuilder;
        private final String name;
        private final Value memory;
        private final Value querySource;
        private final Value validMask;
        private final int attentionHeads;
        private final List<SingleQueryCrossAttentionReadout> readouts;
        private int queryIndex;
        private float epsilon;
        private boolean built;

        private SingleQueryCrossAttentionReadoutGroupBuilder(
                Builder recipeBuilder,
                String name,
                Value memory,
                Value querySource,
                Value validMask,
                int attentionHeads) {
            this.recipeBuilder = recipeBuilder;
            this.name = name;
            this.memory = memory;
            this.querySource = querySource;
            this.validMask = validMask;
            this.attentionHeads = attentionHeads;
            readouts = new ArrayList<>();
            epsilon = DEFAULT_EPSILON;
        }

        /**
         * Adds one purpose-specific readout.
         *
         * <p>Projection weights use {@code [outputWidth, inputWidth]} layout. The key-value weight
         * stores all key rows followed by all value rows. Every projection and bias uses one shared
         * compute data type, which may differ from the memory and query-source types. LayerNorm
         * scale and bias pairs may instead use FLOAT32, but all LayerNorm parameters in the group
         * must share one data type.
         *
         * @param querySeedWeight query-seed projection weight
         * @param querySeedBias query-seed projection bias
         * @param queryWeight attention query projection weight
         * @param queryBias attention query projection bias
         * @param keyValueWeight combined key-value projection weight
         * @param contextWeight attention-context projection weight
         * @param contextBias attention-context projection bias
         * @param queryNormWeight post-attention LayerNorm scale
         * @param queryNormBias post-attention LayerNorm bias
         * @param feedForwardNormWeight feed-forward-input LayerNorm scale
         * @param feedForwardNormBias feed-forward-input LayerNorm bias
         * @param feedForwardExpansionWeight feed-forward expansion weight
         * @param feedForwardExpansionBias feed-forward expansion bias
         * @param feedForwardProjectionWeight feed-forward projection weight
         * @param feedForwardProjectionBias feed-forward projection bias
         * @param outputNormWeight output LayerNorm scale
         * @param outputNormBias output LayerNorm bias
         * @return this builder
         */
        public SingleQueryCrossAttentionReadoutGroupBuilder addReadout(
                Constant querySeedWeight,
                Constant querySeedBias,
                Constant queryWeight,
                Constant queryBias,
                Constant keyValueWeight,
                Constant contextWeight,
                Constant contextBias,
                Constant queryNormWeight,
                Constant queryNormBias,
                Constant feedForwardNormWeight,
                Constant feedForwardNormBias,
                Constant feedForwardExpansionWeight,
                Constant feedForwardExpansionBias,
                Constant feedForwardProjectionWeight,
                Constant feedForwardProjectionBias,
                Constant outputNormWeight,
                Constant outputNormBias) {
            checkMutable();
            Constant[] constants = {
                querySeedWeight,
                querySeedBias,
                queryWeight,
                queryBias,
                keyValueWeight,
                contextWeight,
                contextBias,
                queryNormWeight,
                queryNormBias,
                feedForwardNormWeight,
                feedForwardNormBias,
                feedForwardExpansionWeight,
                feedForwardExpansionBias,
                feedForwardProjectionWeight,
                feedForwardProjectionBias,
                outputNormWeight,
                outputNormBias
            };
            for (Constant constant : constants) {
                recipeBuilder.checkValue(constant);
            }
            readouts.add(
                    new SingleQueryCrossAttentionReadout(
                            querySeedWeight,
                            querySeedBias,
                            queryWeight,
                            queryBias,
                            keyValueWeight,
                            contextWeight,
                            contextBias,
                            queryNormWeight,
                            queryNormBias,
                            feedForwardNormWeight,
                            feedForwardNormBias,
                            feedForwardExpansionWeight,
                            feedForwardExpansionBias,
                            feedForwardProjectionWeight,
                            feedForwardProjectionBias,
                            outputNormWeight,
                            outputNormBias,
                            0));
            return this;
        }

        /**
         * Selects one token from a three-dimensional query source.
         *
         * <p>A two-dimensional query source already contains one query per batch and therefore only
         * accepts index zero. Selection is part of the fused stage and does not materialize a token
         * view.
         *
         * @param queryIndex the zero-based query token index
         * @return this builder
         */
        public SingleQueryCrossAttentionReadoutGroupBuilder optQueryIndex(int queryIndex) {
            checkMutable();
            if (queryIndex < 0) {
                throw new IllegalArgumentException("Readout query index must not be negative.");
            }
            this.queryIndex = queryIndex;
            return this;
        }

        /**
         * Sets the epsilon used by every LayerNorm in the group.
         *
         * @param epsilon the finite positive epsilon
         * @return this builder
         */
        public SingleQueryCrossAttentionReadoutGroupBuilder optEpsilon(float epsilon) {
            checkMutable();
            if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
                throw new IllegalArgumentException(
                        "LayerNorm epsilon must be finite and positive.");
            }
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Adds the immutable readout group to its recipe.
         *
         * @return the single-query readout group value
         */
        public SingleQueryCrossAttentionReadoutGroup build() {
            checkMutable();
            recipeBuilder.checkMutable();
            TensorSpec memorySpec = memory.getSpec();
            TensorSpec querySourceSpec = querySource.getSpec();
            TensorSpec maskSpec = validMask.getSpec();
            if (memorySpec.leadingDimension == null || memorySpec.innerShape.length != 2) {
                throw new IllegalArgumentException(
                        "Readout memory must have shape [bounded batch, tokens, hiddenWidth].");
            }
            if (!Builder.isAffineDataType(memorySpec.dataType)) {
                throw new IllegalArgumentException(
                        "Readout memory only supports FLOAT16, BFLOAT16, and FLOAT32.");
            }
            long tokenCount = memorySpec.innerShape[0];
            long hiddenWidth = memorySpec.innerShape[1];
            boolean scalarQuery =
                    querySourceSpec.innerShape.length == 1
                            && querySourceSpec.innerShape[0] == hiddenWidth;
            boolean sequenceQuery =
                    querySourceSpec.innerShape.length == 2
                            && querySourceSpec.innerShape[0] > 0
                            && querySourceSpec.innerShape[1] == hiddenWidth;
            long queryTokenCount =
                    scalarQuery ? 1 : sequenceQuery ? querySourceSpec.innerShape[0] : 0;
            if (querySourceSpec.leadingDimension != memorySpec.leadingDimension
                    || querySourceSpec.dataType != memorySpec.dataType
                    || queryTokenCount == 0
                    || queryIndex >= queryTokenCount) {
                throw new IllegalArgumentException(
                        "Readout query source must match the memory batch, type, and hidden width,"
                                + " and contain the selected query index.");
            }
            if (maskSpec.leadingDimension != memorySpec.leadingDimension
                    || maskSpec.innerShape.length != 1
                    || maskSpec.innerShape[0] != tokenCount
                    || !Builder.isMaskDataType(maskSpec.dataType)) {
                throw new IllegalArgumentException(
                        "Readout mask must match the memory batch and token count and use a"
                                + " supported mask type.");
            }
            if (attentionHeads <= 0) {
                throw new IllegalArgumentException("Readout attention heads must be positive.");
            }
            if (readouts.isEmpty() || readouts.size() > MAXIMUM_READOUTS) {
                throw new IllegalStateException(
                        "A readout group requires between one and eight readouts.");
            }

            int attentionWidth = -1;
            int maximumFeedForwardWidth = 0;
            DataType normDataType = null;
            DataType computeDataType = readouts.get(0).querySeedWeight.getSpec().dataType;
            if (!Builder.isAffineDataType(computeDataType)) {
                throw new IllegalArgumentException(
                        "Readout projections only support FLOAT16, BFLOAT16, and FLOAT32.");
            }
            List<SingleQueryCrossAttentionReadout> checkedReadouts =
                    new ArrayList<>(readouts.size());
            for (SingleQueryCrossAttentionReadout readout : readouts) {
                requireProjection(
                        readout.querySeedWeight,
                        hiddenWidth,
                        2L * hiddenWidth,
                        computeDataType,
                        "query-seed weight");
                requireVector(
                        readout.querySeedBias, hiddenWidth, computeDataType, "query-seed bias");
                TensorSpec queryWeightSpec = readout.queryWeight.getSpec();
                if (queryWeightSpec.leadingDimension != null
                        || queryWeightSpec.dataType != computeDataType
                        || queryWeightSpec.innerShape.length != 2
                        || queryWeightSpec.innerShape[1] != hiddenWidth) {
                    throw new IllegalArgumentException(
                            "Readout query weight shape or type mismatch.");
                }
                int currentAttentionWidth = Math.toIntExact(queryWeightSpec.innerShape[0]);
                if (attentionWidth < 0) {
                    attentionWidth = currentAttentionWidth;
                } else if (attentionWidth != currentAttentionWidth) {
                    throw new IllegalArgumentException(
                            "All readouts in a group must use one attention width.");
                }
                if (attentionWidth <= 0 || attentionWidth % attentionHeads != 0) {
                    throw new IllegalArgumentException(
                            "Readout attention width must be positive and divisible by its heads.");
                }
                requireVector(readout.queryBias, attentionWidth, computeDataType, "query bias");
                requireProjection(
                        readout.keyValueWeight,
                        2L * attentionWidth,
                        hiddenWidth,
                        computeDataType,
                        "key-value weight");
                requireProjection(
                        readout.contextWeight,
                        hiddenWidth,
                        attentionWidth,
                        computeDataType,
                        "context weight");
                requireVector(readout.contextBias, hiddenWidth, computeDataType, "context bias");
                normDataType =
                        requireNormPair(
                                readout.queryNormWeight,
                                readout.queryNormBias,
                                hiddenWidth,
                                computeDataType,
                                normDataType,
                                "post-attention LayerNorm");
                normDataType =
                        requireNormPair(
                                readout.feedForwardNormWeight,
                                readout.feedForwardNormBias,
                                hiddenWidth,
                                computeDataType,
                                normDataType,
                                "feed-forward LayerNorm");

                TensorSpec expansionSpec = readout.feedForwardExpansionWeight.getSpec();
                if (expansionSpec.leadingDimension != null
                        || expansionSpec.dataType != computeDataType
                        || expansionSpec.innerShape.length != 2
                        || expansionSpec.innerShape[1] != hiddenWidth) {
                    throw new IllegalArgumentException(
                            "Readout feed-forward expansion shape or type mismatch.");
                }
                int feedForwardWidth = Math.toIntExact(expansionSpec.innerShape[0]);
                requireVector(
                        readout.feedForwardExpansionBias,
                        feedForwardWidth,
                        computeDataType,
                        "feed-forward expansion bias");
                requireProjection(
                        readout.feedForwardProjectionWeight,
                        hiddenWidth,
                        feedForwardWidth,
                        computeDataType,
                        "feed-forward projection weight");
                requireVector(
                        readout.feedForwardProjectionBias,
                        hiddenWidth,
                        computeDataType,
                        "feed-forward projection bias");
                normDataType =
                        requireNormPair(
                                readout.outputNormWeight,
                                readout.outputNormBias,
                                hiddenWidth,
                                computeDataType,
                                normDataType,
                                "output LayerNorm");
                maximumFeedForwardWidth = Math.max(maximumFeedForwardWidth, feedForwardWidth);
                checkedReadouts.add(
                        new SingleQueryCrossAttentionReadout(
                                readout.querySeedWeight,
                                readout.querySeedBias,
                                readout.queryWeight,
                                readout.queryBias,
                                readout.keyValueWeight,
                                readout.contextWeight,
                                readout.contextBias,
                                readout.queryNormWeight,
                                readout.queryNormBias,
                                readout.feedForwardNormWeight,
                                readout.feedForwardNormBias,
                                readout.feedForwardExpansionWeight,
                                readout.feedForwardExpansionBias,
                                readout.feedForwardProjectionWeight,
                                readout.feedForwardProjectionBias,
                                readout.outputNormWeight,
                                readout.outputNormBias,
                                feedForwardWidth));
            }

            List<String> stateNames = new ArrayList<>(readouts.size());
            for (int index = 0; index < readouts.size(); ++index) {
                stateNames.add(recipeBuilder.addValueName(name + '[' + index + ']'));
            }
            SingleQueryCrossAttentionReadoutGroup value =
                    new SingleQueryCrossAttentionReadoutGroup(
                            recipeBuilder.owner,
                            recipeBuilder.values.size(),
                            stateNames,
                            TensorSpec.of(
                                    computeDataType, memorySpec.leadingDimension, hiddenWidth),
                            memory,
                            querySource,
                            validMask,
                            queryIndex,
                            attentionHeads,
                            attentionWidth,
                            maximumFeedForwardWidth,
                            epsilon,
                            checkedReadouts);
            recipeBuilder.values.addAll(value.states);
            built = true;
            return value;
        }

        private static void requireProjection(
                Constant constant, long rows, long columns, DataType dataType, String name) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 2
                    || spec.innerShape[0] != rows
                    || spec.innerShape[1] != columns) {
                throw new IllegalArgumentException("Readout " + name + " shape or type mismatch.");
            }
        }

        private static void requireVector(
                Constant constant, long width, DataType dataType, String name) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 1
                    || spec.innerShape[0] != width) {
                throw new IllegalArgumentException("Readout " + name + " shape or type mismatch.");
            }
        }

        private static DataType requireNormPair(
                Constant weight,
                Constant bias,
                long width,
                DataType valueDataType,
                DataType groupNormDataType,
                String name) {
            TensorSpec weightSpec = weight.getSpec();
            TensorSpec biasSpec = bias.getSpec();
            DataType dataType = weightSpec.dataType;
            if (weightSpec.leadingDimension != null
                    || biasSpec.leadingDimension != null
                    || weightSpec.innerShape.length != 1
                    || biasSpec.innerShape.length != 1
                    || weightSpec.innerShape[0] != width
                    || biasSpec.innerShape[0] != width
                    || biasSpec.dataType != dataType
                    || (dataType != valueDataType && dataType != DataType.FLOAT32)) {
                throw new IllegalArgumentException("Readout " + name + " shape or type mismatch.");
            }
            if (groupNormDataType != null && groupNormDataType != dataType) {
                throw new IllegalArgumentException(
                        "All LayerNorm parameters in a readout group must use one data type.");
            }
            return dataType;
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException("The single-query readout group has been built.");
            }
        }
    }

    /** Builds one {@link IndexedLocalTransformerEncoder} value within a {@link Builder}. */
    public static final class IndexedLocalTransformerEncoderBuilder {

        private static final float DEFAULT_EPSILON = 1.0e-5f;

        private final Builder recipeBuilder;
        private final String name;
        private final List<Value> inputSegments;
        private final Value indices;
        private final int attentionHeads;
        private final int attentionWidth;
        private final int feedForwardWidth;
        private Constant inputNormWeight;
        private Constant inputNormBias;
        private TransformerEncoderBlock block;
        private float epsilon;
        private boolean built;

        private IndexedLocalTransformerEncoderBuilder(
                Builder recipeBuilder,
                String name,
                List<Value> inputSegments,
                Value indices,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth) {
            this.recipeBuilder = recipeBuilder;
            this.name = name;
            this.inputSegments = immutableCopy(inputSegments);
            this.indices = indices;
            this.attentionHeads = attentionHeads;
            this.attentionWidth = attentionWidth;
            this.feedForwardWidth = feedForwardWidth;
            epsilon = DEFAULT_EPSILON;
        }

        /**
         * Sets the LayerNorm applied immediately after selected rows are gathered.
         *
         * @param weight the input LayerNorm scale
         * @param bias the input LayerNorm bias
         * @return this builder
         */
        public IndexedLocalTransformerEncoderBuilder setInputNormalization(
                Constant weight, Constant bias) {
            checkMutable();
            recipeBuilder.checkValue(weight);
            recipeBuilder.checkValue(bias);
            inputNormWeight = weight;
            inputNormBias = bias;
            return this;
        }

        /**
         * Sets the pre-normalized attention and SiLU feed-forward block.
         *
         * @param attentionInputWeight attention-input LayerNorm scale
         * @param attentionInputBias attention-input LayerNorm bias
         * @param queryKeyValueWeight combined QKV projection weight
         * @param attentionOutputWeight attention output projection weight
         * @param attentionOutputBias attention output projection bias
         * @param feedForwardInputWeight feed-forward-input LayerNorm scale
         * @param feedForwardInputBias feed-forward-input LayerNorm bias
         * @param feedForwardExpansionWeight feed-forward expansion weight
         * @param feedForwardExpansionBias feed-forward expansion bias
         * @param feedForwardProjectionWeight feed-forward projection weight
         * @param feedForwardProjectionBias feed-forward projection bias
         * @param outputWeight output LayerNorm scale
         * @param outputBias output LayerNorm bias
         * @return this builder
         */
        public IndexedLocalTransformerEncoderBuilder setBlock(
                Constant attentionInputWeight,
                Constant attentionInputBias,
                Constant queryKeyValueWeight,
                Constant attentionOutputWeight,
                Constant attentionOutputBias,
                Constant feedForwardInputWeight,
                Constant feedForwardInputBias,
                Constant feedForwardExpansionWeight,
                Constant feedForwardExpansionBias,
                Constant feedForwardProjectionWeight,
                Constant feedForwardProjectionBias,
                Constant outputWeight,
                Constant outputBias) {
            checkMutable();
            Constant[] constants = {
                attentionInputWeight,
                attentionInputBias,
                queryKeyValueWeight,
                attentionOutputWeight,
                attentionOutputBias,
                feedForwardInputWeight,
                feedForwardInputBias,
                feedForwardExpansionWeight,
                feedForwardExpansionBias,
                feedForwardProjectionWeight,
                feedForwardProjectionBias,
                outputWeight,
                outputBias
            };
            for (Constant constant : constants) {
                recipeBuilder.checkValue(constant);
            }
            block =
                    new TransformerEncoderBlock(
                            attentionInputWeight,
                            attentionInputBias,
                            queryKeyValueWeight,
                            attentionOutputWeight,
                            attentionOutputBias,
                            feedForwardInputWeight,
                            feedForwardInputBias,
                            feedForwardExpansionWeight,
                            feedForwardExpansionBias,
                            feedForwardProjectionWeight,
                            feedForwardProjectionBias,
                            outputWeight,
                            outputBias);
            return this;
        }

        /**
         * Sets the epsilon used by every LayerNorm in this encoder.
         *
         * @param epsilon the finite positive epsilon
         * @return this builder
         */
        public IndexedLocalTransformerEncoderBuilder optEpsilon(float epsilon) {
            checkMutable();
            if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
                throw new IllegalArgumentException(
                        "LayerNorm epsilon must be finite and positive.");
            }
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Adds the indexed local transformer encoder to its recipe.
         *
         * @return the indexed local transformer encoder value
         */
        public IndexedLocalTransformerEncoder build() {
            checkMutable();
            recipeBuilder.checkMutable();
            TensorSpec inputSpec = inputSegments.get(0).getSpec();
            TensorSpec indexSpec = indices.getSpec();
            if (inputSpec.leadingDimension == null || inputSpec.innerShape.length != 3) {
                throw new IllegalArgumentException(
                        "Indexed local transformer input must have shape "
                                + "[bounded batch, groups, tokens, hiddenWidth].");
            }
            if (!Builder.isAffineDataType(inputSpec.dataType)) {
                throw new IllegalArgumentException(
                        "Indexed local transformer input must be floating point.");
            }
            if (indexSpec.leadingDimension == null
                    || indexSpec.innerShape.length != 0
                    || (indexSpec.dataType != DataType.INT32
                            && indexSpec.dataType != DataType.INT64)) {
                throw new IllegalArgumentException(
                        "Indexed local transformer indices must be bounded INT32 or INT64 rows.");
            }
            long groupCount = inputSpec.innerShape[0];
            long tokenCount = 0;
            long hiddenWidth = inputSpec.innerShape[2];
            for (Value inputSegment : inputSegments) {
                TensorSpec segmentSpec = inputSegment.getSpec();
                if (segmentSpec.leadingDimension != inputSpec.leadingDimension
                        || segmentSpec.dataType != inputSpec.dataType
                        || segmentSpec.innerShape.length != 3
                        || segmentSpec.innerShape[0] != groupCount
                        || segmentSpec.innerShape[1] <= 0
                        || segmentSpec.innerShape[2] != hiddenWidth) {
                    throw new IllegalArgumentException(
                            "Indexed local transformer input segments must share batch, groups,"
                                + " hidden width, and data type and contain positive token"
                                + " counts.");
                }
                tokenCount = Math.addExact(tokenCount, segmentSpec.innerShape[1]);
            }
            long maximumDenseRows =
                    Math.multiplyExact(
                            inputSpec.leadingDimension.maximumExtent,
                            Math.multiplyExact(groupCount, tokenCount));
            if (indexSpec.leadingDimension.maximumExtent > maximumDenseRows) {
                throw new IllegalArgumentException(
                        "Indexed local transformer index capacity exceeds the dense input.");
            }
            if (inputSpec.innerShape[0] <= 0
                    || tokenCount <= 0
                    || attentionHeads <= 0
                    || attentionWidth <= 0
                    || attentionWidth % attentionHeads != 0
                    || feedForwardWidth <= 0) {
                throw new IllegalArgumentException(
                        "Indexed local transformer dimensions must be positive and divisible.");
            }
            if (inputNormWeight == null || inputNormBias == null || block == null) {
                throw new IllegalStateException(
                        "Indexed local transformer normalization and block must be set.");
            }
            requireNormPair(
                    inputNormWeight,
                    inputNormBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "input LayerNorm");
            requireNormPair(
                    block.attentionInputWeight,
                    block.attentionInputBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "attention-input LayerNorm");
            requireProjection(
                    block.queryKeyValueWeight,
                    3L * attentionWidth,
                    hiddenWidth,
                    inputSpec.dataType,
                    "QKV weight");
            requireProjection(
                    block.attentionOutputWeight,
                    hiddenWidth,
                    attentionWidth,
                    inputSpec.dataType,
                    "attention output weight");
            requireVector(
                    block.attentionOutputBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "attention output bias");
            requireNormPair(
                    block.feedForwardInputWeight,
                    block.feedForwardInputBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "feed-forward-input LayerNorm");
            requireProjection(
                    block.feedForwardExpansionWeight,
                    feedForwardWidth,
                    hiddenWidth,
                    inputSpec.dataType,
                    "feed-forward expansion weight");
            requireVector(
                    block.feedForwardExpansionBias,
                    feedForwardWidth,
                    inputSpec.dataType,
                    "feed-forward expansion bias");
            requireProjection(
                    block.feedForwardProjectionWeight,
                    hiddenWidth,
                    feedForwardWidth,
                    inputSpec.dataType,
                    "feed-forward projection weight");
            requireVector(
                    block.feedForwardProjectionBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "feed-forward projection bias");
            requireNormPair(
                    block.outputWeight,
                    block.outputBias,
                    hiddenWidth,
                    inputSpec.dataType,
                    "output LayerNorm");

            String checkedName = recipeBuilder.addValueName(name);
            IndexedLocalTransformerEncoder value =
                    new IndexedLocalTransformerEncoder(
                            recipeBuilder.owner,
                            recipeBuilder.values.size(),
                            checkedName,
                            TensorSpec.of(
                                    inputSpec.dataType,
                                    inputSpec.leadingDimension,
                                    groupCount,
                                    tokenCount,
                                    hiddenWidth),
                            inputSegments,
                            indices,
                            inputNormWeight,
                            inputNormBias,
                            block,
                            attentionHeads,
                            attentionWidth,
                            feedForwardWidth,
                            epsilon);
            recipeBuilder.values.add(value);
            built = true;
            return value;
        }

        private static void requireProjection(
                Constant constant, long rows, long columns, DataType dataType, String name) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 2
                    || spec.innerShape[0] != rows
                    || spec.innerShape[1] != columns) {
                throw new IllegalArgumentException(
                        "Indexed local transformer " + name + " shape or type mismatch.");
            }
        }

        private static void requireVector(
                Constant constant, long width, DataType dataType, String name) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 1
                    || spec.innerShape[0] != width) {
                throw new IllegalArgumentException(
                        "Indexed local transformer " + name + " shape or type mismatch.");
            }
        }

        private static void requireNormPair(
                Constant weight, Constant bias, long width, DataType valueDataType, String name) {
            TensorSpec weightSpec = weight.getSpec();
            TensorSpec biasSpec = bias.getSpec();
            if (weightSpec.leadingDimension != null
                    || biasSpec.leadingDimension != null
                    || weightSpec.innerShape.length != 1
                    || biasSpec.innerShape.length != 1
                    || weightSpec.innerShape[0] != width
                    || biasSpec.innerShape[0] != width
                    || weightSpec.dataType != biasSpec.dataType
                    || (weightSpec.dataType != valueDataType
                            && weightSpec.dataType != DataType.FLOAT32)) {
                throw new IllegalArgumentException(
                        "Indexed local transformer " + name + " shape or type mismatch.");
            }
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException(
                        "The indexed local transformer encoder has been built.");
            }
        }
    }

    /** Builds one {@link TransformerEncoderStack} value within a {@link Builder}. */
    public static final class TransformerEncoderStackBuilder {

        private static final float DEFAULT_EPSILON = 1.0e-5f;

        private final Builder recipeBuilder;
        private final String name;
        private final Value input;
        private final int attentionHeads;
        private final int attentionWidth;
        private final int feedForwardWidth;
        private final List<TransformerEncoderBlock> blocks;
        private float epsilon;
        private boolean built;

        private TransformerEncoderStackBuilder(
                Builder recipeBuilder,
                String name,
                Value input,
                int attentionHeads,
                int attentionWidth,
                int feedForwardWidth) {
            this.recipeBuilder = recipeBuilder;
            this.name = name;
            this.input = input;
            this.attentionHeads = attentionHeads;
            this.attentionWidth = attentionWidth;
            this.feedForwardWidth = feedForwardWidth;
            blocks = new ArrayList<>();
            epsilon = DEFAULT_EPSILON;
        }

        /**
         * Adds one pre-normalized attention and SiLU feed-forward block.
         *
         * @param attentionInputWeight attention-input LayerNorm scale
         * @param attentionInputBias attention-input LayerNorm bias
         * @param queryKeyValueWeight combined QKV projection weight
         * @param attentionOutputWeight attention output projection weight
         * @param attentionOutputBias attention output projection bias
         * @param feedForwardInputWeight feed-forward-input LayerNorm scale
         * @param feedForwardInputBias feed-forward-input LayerNorm bias
         * @param feedForwardExpansionWeight feed-forward expansion weight
         * @param feedForwardExpansionBias feed-forward expansion bias
         * @param feedForwardProjectionWeight feed-forward projection weight
         * @param feedForwardProjectionBias feed-forward projection bias
         * @param outputWeight output LayerNorm scale
         * @param outputBias output LayerNorm bias
         * @return this builder
         */
        public TransformerEncoderStackBuilder addBlock(
                Constant attentionInputWeight,
                Constant attentionInputBias,
                Constant queryKeyValueWeight,
                Constant attentionOutputWeight,
                Constant attentionOutputBias,
                Constant feedForwardInputWeight,
                Constant feedForwardInputBias,
                Constant feedForwardExpansionWeight,
                Constant feedForwardExpansionBias,
                Constant feedForwardProjectionWeight,
                Constant feedForwardProjectionBias,
                Constant outputWeight,
                Constant outputBias) {
            checkMutable();
            Constant[] constants = {
                attentionInputWeight,
                attentionInputBias,
                queryKeyValueWeight,
                attentionOutputWeight,
                attentionOutputBias,
                feedForwardInputWeight,
                feedForwardInputBias,
                feedForwardExpansionWeight,
                feedForwardExpansionBias,
                feedForwardProjectionWeight,
                feedForwardProjectionBias,
                outputWeight,
                outputBias
            };
            for (Constant constant : constants) {
                recipeBuilder.checkValue(constant);
            }
            blocks.add(
                    new TransformerEncoderBlock(
                            attentionInputWeight,
                            attentionInputBias,
                            queryKeyValueWeight,
                            attentionOutputWeight,
                            attentionOutputBias,
                            feedForwardInputWeight,
                            feedForwardInputBias,
                            feedForwardExpansionWeight,
                            feedForwardExpansionBias,
                            feedForwardProjectionWeight,
                            feedForwardProjectionBias,
                            outputWeight,
                            outputBias));
            return this;
        }

        /**
         * Sets the epsilon used by every LayerNorm in this stack.
         *
         * @param epsilon the finite positive epsilon
         * @return this builder
         */
        public TransformerEncoderStackBuilder optEpsilon(float epsilon) {
            checkMutable();
            if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
                throw new IllegalArgumentException(
                        "LayerNorm epsilon must be finite and positive.");
            }
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Adds the immutable transformer encoder stack to its recipe.
         *
         * @return the transformer encoder stack value
         */
        public TransformerEncoderStack build() {
            checkMutable();
            recipeBuilder.checkMutable();
            TensorSpec inputSpec = input.getSpec();
            if (inputSpec.leadingDimension == null || inputSpec.innerShape.length != 2) {
                throw new IllegalArgumentException(
                        "Transformer input must have shape [bounded batch, tokens, hiddenWidth].");
            }
            if (!Builder.isAffineDataType(inputSpec.dataType)) {
                throw new IllegalArgumentException(
                        "Transformer input only supports FLOAT16, BFLOAT16, and FLOAT32.");
            }
            long hiddenWidth = inputSpec.innerShape[1];
            if (attentionHeads <= 0
                    || attentionWidth <= 0
                    || attentionWidth % attentionHeads != 0
                    || feedForwardWidth <= 0) {
                throw new IllegalArgumentException(
                        "Transformer widths and head count must be positive and divisible.");
            }
            if (blocks.isEmpty()) {
                throw new IllegalStateException("A transformer encoder stack requires a block.");
            }
            for (TransformerEncoderBlock block : blocks) {
                requireNorm(block.attentionInputWeight, hiddenWidth, inputSpec.dataType);
                requireNorm(block.attentionInputBias, hiddenWidth, inputSpec.dataType);
                requireSameDataType(block.attentionInputWeight, block.attentionInputBias);
                requireProjection(
                        block.queryKeyValueWeight,
                        3L * attentionWidth,
                        hiddenWidth,
                        inputSpec.dataType);
                requireProjection(
                        block.attentionOutputWeight,
                        hiddenWidth,
                        attentionWidth,
                        inputSpec.dataType);
                requireVector(block.attentionOutputBias, hiddenWidth, inputSpec.dataType);
                requireNorm(block.feedForwardInputWeight, hiddenWidth, inputSpec.dataType);
                requireNorm(block.feedForwardInputBias, hiddenWidth, inputSpec.dataType);
                requireSameDataType(block.feedForwardInputWeight, block.feedForwardInputBias);
                requireProjection(
                        block.feedForwardExpansionWeight,
                        feedForwardWidth,
                        hiddenWidth,
                        inputSpec.dataType);
                requireVector(block.feedForwardExpansionBias, feedForwardWidth, inputSpec.dataType);
                requireProjection(
                        block.feedForwardProjectionWeight,
                        hiddenWidth,
                        feedForwardWidth,
                        inputSpec.dataType);
                requireVector(block.feedForwardProjectionBias, hiddenWidth, inputSpec.dataType);
                requireNorm(block.outputWeight, hiddenWidth, inputSpec.dataType);
                requireNorm(block.outputBias, hiddenWidth, inputSpec.dataType);
                requireSameDataType(block.outputWeight, block.outputBias);
            }

            String checkedName = recipeBuilder.addValueName(name);
            TransformerEncoderStack value =
                    new TransformerEncoderStack(
                            recipeBuilder.owner,
                            recipeBuilder.values.size(),
                            checkedName,
                            TensorSpec.of(
                                    inputSpec.dataType,
                                    inputSpec.leadingDimension,
                                    inputSpec.innerShape),
                            input,
                            attentionHeads,
                            attentionWidth,
                            feedForwardWidth,
                            epsilon,
                            blocks);
            recipeBuilder.values.add(value);
            built = true;
            return value;
        }

        private static void requireProjection(
                Constant constant, long rows, long columns, DataType dataType) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 2
                    || spec.innerShape[0] != rows
                    || spec.innerShape[1] != columns) {
                throw new IllegalArgumentException(
                        "Transformer projection shape or type mismatch.");
            }
        }

        private static void requireVector(Constant constant, long width, DataType dataType) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || spec.dataType != dataType
                    || spec.innerShape.length != 1
                    || spec.innerShape[0] != width) {
                throw new IllegalArgumentException("Transformer bias shape or type mismatch.");
            }
        }

        private static void requireNorm(Constant constant, long width, DataType dataType) {
            TensorSpec spec = constant.getSpec();
            if (spec.leadingDimension != null
                    || (spec.dataType != dataType && spec.dataType != DataType.FLOAT32)
                    || spec.innerShape.length != 1
                    || spec.innerShape[0] != width) {
                throw new IllegalArgumentException(
                        "Transformer LayerNorm parameter shape or type mismatch.");
            }
        }

        private static void requireSameDataType(Constant left, Constant right) {
            if (left.getSpec().dataType != right.getSpec().dataType) {
                throw new IllegalArgumentException(
                        "Transformer LayerNorm scale and bias must use one data type.");
            }
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException("The transformer encoder stack has been built.");
            }
        }
    }

    /** Builds one {@link AffineSum} value within a {@link Builder}. */
    public static final class AffineSumBuilder {

        private final Builder recipeBuilder;
        private final String name;
        private final long outputWidth;
        private final List<AffineTerm> terms;
        private Constant bias;
        private Activation activation;
        private boolean built;

        private AffineSumBuilder(Builder recipeBuilder, String name, long outputWidth) {
            this.recipeBuilder = recipeBuilder;
            this.name = name;
            this.outputWidth = outputWidth;
            terms = new ArrayList<>();
            activation = Activation.NONE;
        }

        /**
         * Adds a projected value to the sum.
         *
         * @param input a FLOAT16, BFLOAT16, or FLOAT32 dynamic value, or a fixed value whose
         *     singleton first axis is included in the output rank, whose last axis is the input
         *     feature width
         * @param weight a fixed {@code [outputWidth, inputWidth]} constant whose data type selects
         *     the common projection data type
         * @return this builder
         */
        public AffineSumBuilder addTerm(Value input, Constant weight) {
            checkMutable();
            recipeBuilder.checkValue(input);
            recipeBuilder.checkValue(weight);
            terms.add(new AffineTerm(input, weight));
            return this;
        }

        /**
         * Sets the optional one-dimensional bias.
         *
         * @param bias a fixed {@code [outputWidth]} constant
         * @return this builder
         */
        public AffineSumBuilder optBias(Constant bias) {
            checkMutable();
            recipeBuilder.checkValue(bias);
            this.bias = bias;
            return this;
        }

        /**
         * Sets the activation applied after all terms and the bias are added.
         *
         * @param activation the activation
         * @return this builder
         */
        public AffineSumBuilder optActivation(Activation activation) {
            checkMutable();
            this.activation = Objects.requireNonNull(activation, "activation");
            return this;
        }

        /**
         * Adds the immutable affine-sum value to its recipe.
         *
         * @return the affine-sum value
         */
        public AffineSum build() {
            checkMutable();
            recipeBuilder.checkMutable();
            if (terms.isEmpty()) {
                throw new IllegalStateException("An affine sum requires at least one term.");
            }

            DataType projectionDataType = null;
            Dimension leadingDimension = null;
            long[] outputPrefix = new long[0];
            for (AffineTerm term : terms) {
                TensorSpec inputSpec = term.input.spec;
                TensorSpec weightSpec = term.weight.getSpec();
                boolean dynamicLeading = inputSpec.leadingDimension != null;
                if (inputSpec.innerShape.length == 0
                        || (!dynamicLeading
                                && (inputSpec.innerShape.length < 2
                                        || inputSpec.innerShape[0] != 1))) {
                    throw new IllegalArgumentException(
                            "Fixed affine inputs require a singleton leading axis and feature"
                                    + " axis.");
                }
                if (!Builder.isAffineDataType(inputSpec.dataType)) {
                    throw new IllegalArgumentException(
                            "Affine sources only support FLOAT16, BFLOAT16, and FLOAT32.");
                }
                if (dynamicLeading) {
                    if (leadingDimension == null) {
                        leadingDimension = inputSpec.leadingDimension;
                    } else if (leadingDimension != inputSpec.leadingDimension) {
                        throw new IllegalArgumentException(
                                "Dynamic affine inputs must share the same leading dimension.");
                    }
                }
                if (weightSpec.leadingDimension != null || weightSpec.innerShape.length != 2) {
                    throw new IllegalArgumentException(
                            "Affine weights must have a fixed two-dimensional shape.");
                }
                if (!Builder.isAffineDataType(weightSpec.dataType)) {
                    throw new IllegalArgumentException(
                            "Affine weights only support FLOAT16, BFLOAT16, and FLOAT32.");
                }
                if (projectionDataType == null) {
                    projectionDataType = weightSpec.dataType;
                } else if (projectionDataType != weightSpec.dataType) {
                    throw new IllegalArgumentException(
                            "Affine weights must use one projection data type.");
                }
                if (weightSpec.innerShape[0] != outputWidth
                        || weightSpec.innerShape[1]
                                != inputSpec.innerShape[inputSpec.innerShape.length - 1]) {
                    throw new IllegalArgumentException(
                            "Affine weight shape does not match its term.");
                }
                outputPrefix =
                        broadcastShape(
                                outputPrefix,
                                inputSpec.innerShape,
                                dynamicLeading ? 0 : 1,
                                inputSpec.innerShape.length - (dynamicLeading ? 1 : 2));
            }
            if (leadingDimension == null) {
                throw new IllegalArgumentException(
                        "An affine sum requires at least one dynamically bounded input.");
            }
            for (AffineTerm term : terms) {
                TensorSpec inputSpec = term.input.spec;
                if (inputSpec.leadingDimension == null
                        && inputSpec.innerShape.length != outputPrefix.length + 2) {
                    throw new IllegalArgumentException(
                            "Fixed affine inputs must match the output rank.");
                }
            }

            if (bias != null) {
                TensorSpec biasSpec = bias.getSpec();
                if (biasSpec.leadingDimension != null
                        || biasSpec.dataType != projectionDataType
                        || biasSpec.innerShape.length != 1
                        || biasSpec.innerShape[0] != outputWidth) {
                    throw new IllegalArgumentException(
                            "Affine bias must have fixed shape [outputWidth] and matching data"
                                    + " type.");
                }
            }

            long[] outputInnerShape = new long[outputPrefix.length + 1];
            System.arraycopy(outputPrefix, 0, outputInnerShape, 0, outputPrefix.length);
            outputInnerShape[outputPrefix.length] = outputWidth;
            String checkedName = recipeBuilder.addValueName(name);
            AffineSum value =
                    new AffineSum(
                            recipeBuilder.owner,
                            recipeBuilder.values.size(),
                            checkedName,
                            TensorSpec.of(projectionDataType, leadingDimension, outputInnerShape),
                            terms,
                            bias,
                            activation);
            recipeBuilder.values.add(value);
            built = true;
            return value;
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException("The affine sum has already been built.");
            }
        }

        private static long[] broadcastShape(
                long[] left, long[] right, int rightOffset, int rightLength) {
            int resultLength = Math.max(left.length, rightLength);
            long[] result = new long[resultLength];
            for (int resultAxis = resultLength - 1; resultAxis >= 0; --resultAxis) {
                int leftAxis = resultAxis - (resultLength - left.length);
                int rightAxis = resultAxis - (resultLength - rightLength);
                long leftExtent = leftAxis < 0 ? 1 : left[leftAxis];
                long rightExtent = rightAxis < 0 ? 1 : right[rightOffset + rightAxis];
                if (leftExtent != rightExtent && leftExtent != 1 && rightExtent != 1) {
                    throw new IllegalArgumentException(
                            "Affine input prefix dimensions are not broadcast-compatible.");
                }
                result[resultAxis] = Math.max(leftExtent, rightExtent);
            }
            return result;
        }
    }

    /** Builds one {@link IndexedAffine} value within a {@link Builder}. */
    public static final class IndexedAffineBuilder {

        private final Builder recipeBuilder;
        private final String name;
        private final Value indices;
        private final Dimension destinationDimension;
        private final List<IndexedAffineSource> sources;
        private Constant hiddenWeight;
        private Constant hiddenBias;
        private Activation activation;
        private Constant outputWeight;
        private Constant outputBias;
        private boolean built;

        private IndexedAffineBuilder(
                Builder recipeBuilder, String name, Value indices, Dimension destinationDimension) {
            this.recipeBuilder = recipeBuilder;
            this.name = name;
            this.indices = indices;
            this.destinationDimension = destinationDimension;
            sources = new ArrayList<>();
            activation = Activation.NONE;
        }

        /**
         * Adds a source gathered with the common destination indices.
         *
         * @param input a two-dimensional, dynamically bounded floating-point value
         * @param indexDivisor the positive divisor applied to destination indices before gathering
         * @return this builder
         */
        public IndexedAffineBuilder addSource(Value input, long indexDivisor) {
            checkMutable();
            recipeBuilder.checkValue(input);
            if (indexDivisor <= 0) {
                throw new IllegalArgumentException("The index divisor must be positive.");
            }
            sources.add(new IndexedAffineSource(input, indexDivisor));
            return this;
        }

        /**
         * Sets the required hidden projection weight.
         *
         * @param weight a fixed {@code [hiddenWidth, concatenatedWidth]} constant
         * @return this builder
         */
        public IndexedAffineBuilder setHiddenWeight(Constant weight) {
            checkMutable();
            recipeBuilder.checkValue(weight);
            hiddenWeight = weight;
            return this;
        }

        /**
         * Sets the optional hidden projection bias.
         *
         * @param bias a fixed {@code [hiddenWidth]} constant
         * @return this builder
         */
        public IndexedAffineBuilder optHiddenBias(Constant bias) {
            checkMutable();
            recipeBuilder.checkValue(bias);
            hiddenBias = bias;
            return this;
        }

        /**
         * Sets the activation applied after the hidden projection.
         *
         * @param activation the hidden activation
         * @return this builder
         */
        public IndexedAffineBuilder optActivation(Activation activation) {
            checkMutable();
            this.activation = Objects.requireNonNull(activation, "activation");
            return this;
        }

        /**
         * Sets the required output projection weight.
         *
         * @param weight a fixed {@code [outputWidth, hiddenWidth]} constant
         * @return this builder
         */
        public IndexedAffineBuilder setOutputWeight(Constant weight) {
            checkMutable();
            recipeBuilder.checkValue(weight);
            outputWeight = weight;
            return this;
        }

        /**
         * Sets the optional output projection bias.
         *
         * @param bias a fixed {@code [outputWidth]} constant
         * @return this builder
         */
        public IndexedAffineBuilder optOutputBias(Constant bias) {
            checkMutable();
            recipeBuilder.checkValue(bias);
            outputBias = bias;
            return this;
        }

        /**
         * Adds the immutable indexed-affine value to its recipe.
         *
         * @return the indexed-affine value
         */
        public IndexedAffine build() {
            checkMutable();
            recipeBuilder.checkMutable();
            TensorSpec indexSpec = indices.spec;
            if (indexSpec.leadingDimension == null
                    || indexSpec.innerShape.length != 0
                    || (indexSpec.dataType != DataType.INT32
                            && indexSpec.dataType != DataType.INT64)) {
                throw new IllegalArgumentException(
                        "Indexed affine indices must be a dynamically bounded INT32 or INT64"
                                + " vector.");
            }
            if (indexSpec.leadingDimension.maximumExtent > destinationDimension.maximumExtent) {
                throw new IllegalArgumentException(
                        "Indexed affine active capacity must not exceed destination capacity.");
            }
            if (sources.isEmpty()) {
                throw new IllegalStateException("An indexed affine requires at least one source.");
            }
            if (hiddenWeight == null || outputWeight == null) {
                throw new IllegalStateException(
                        "An indexed affine requires hidden and output weights.");
            }

            long inputWidth = 0;
            for (IndexedAffineSource source : sources) {
                TensorSpec sourceSpec = source.input.spec;
                if (sourceSpec.leadingDimension == null || sourceSpec.innerShape.length != 1) {
                    throw new IllegalArgumentException(
                            "Indexed affine sources must have one leading and one feature"
                                    + " dimension.");
                }
                if (!Builder.isAffineDataType(sourceSpec.dataType)) {
                    throw new IllegalArgumentException(
                            "Indexed affine sources only support FLOAT16, BFLOAT16, and FLOAT32.");
                }
                long requiredSourceRows =
                        (destinationDimension.maximumExtent - 1) / source.indexDivisor + 1;
                if (sourceSpec.leadingDimension.maximumExtent < requiredSourceRows) {
                    throw new IllegalArgumentException(
                            "Indexed affine source capacity does not cover divided destination"
                                    + " indices.");
                }
                inputWidth = Math.addExact(inputWidth, sourceSpec.innerShape[0]);
            }

            TensorSpec hiddenWeightSpec = hiddenWeight.getSpec();
            if (hiddenWeightSpec.leadingDimension != null
                    || hiddenWeightSpec.innerShape.length != 2
                    || !Builder.isAffineDataType(hiddenWeightSpec.dataType)
                    || hiddenWeightSpec.innerShape[1] != inputWidth) {
                throw new IllegalArgumentException(
                        "Indexed affine hidden weight must have fixed shape"
                                + " [hiddenWidth, concatenatedWidth].");
            }
            long hiddenWidth = hiddenWeightSpec.innerShape[0];
            DataType projectionDataType = hiddenWeightSpec.dataType;
            checkBias(hiddenBias, hiddenWidth, projectionDataType, "hidden");

            TensorSpec outputWeightSpec = outputWeight.getSpec();
            if (outputWeightSpec.leadingDimension != null
                    || outputWeightSpec.innerShape.length != 2
                    || outputWeightSpec.dataType != projectionDataType
                    || outputWeightSpec.innerShape[1] != hiddenWidth) {
                throw new IllegalArgumentException(
                        "Indexed affine output weight must have fixed shape"
                                + " [outputWidth, hiddenWidth] and matching data type.");
            }
            long outputWidth = outputWeightSpec.innerShape[0];
            checkBias(outputBias, outputWidth, projectionDataType, "output");

            String checkedName = recipeBuilder.addValueName(name);
            IndexedAffine value =
                    new IndexedAffine(
                            recipeBuilder.owner,
                            recipeBuilder.values.size(),
                            checkedName,
                            TensorSpec.of(projectionDataType, destinationDimension, outputWidth),
                            indices,
                            sources,
                            hiddenWeight,
                            hiddenBias,
                            activation,
                            outputWeight,
                            outputBias);
            recipeBuilder.values.add(value);
            built = true;
            return value;
        }

        private static void checkBias(Constant bias, long width, DataType dataType, String kind) {
            if (bias == null) {
                return;
            }
            TensorSpec spec = bias.getSpec();
            if (spec.leadingDimension != null
                    || spec.innerShape.length != 1
                    || spec.dataType != dataType
                    || spec.innerShape[0] != width) {
                throw new IllegalArgumentException(
                        "Indexed affine "
                                + kind
                                + " bias must have fixed matching shape and data type.");
            }
        }

        private void checkMutable() {
            if (built) {
                throw new IllegalStateException("The indexed affine has already been built.");
            }
        }
    }
}
