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
 * <p>The initial API supports {@link OutputPack}, which concatenates two-dimensional floating-point
 * values along their last axis and converts them into a contiguous {@link DataType#FLOAT32} value.
 * Additional value types can be added without changing the lifecycle of prepared plans and
 * sessions.
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
                if (!isPackDataType(spec.dataType)) {
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

        private static boolean isPackDataType(DataType dataType) {
            return dataType == DataType.FLOAT16
                    || dataType == DataType.BFLOAT16
                    || dataType == DataType.FLOAT32;
        }

        private static String requireName(String name, String kind) {
            Objects.requireNonNull(name, kind + " name");
            if (name.trim().isEmpty()) {
                throw new IllegalArgumentException("The " + kind + " name must not be empty.");
            }
            return name;
        }
    }
}
