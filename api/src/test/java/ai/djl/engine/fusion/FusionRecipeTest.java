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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class FusionRecipeTest {

    @Test
    public void outputPackBuildsStableTypedGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("mixed-output-pack");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 32);
        FusionRecipe.Input half =
                builder.addInput("half", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 3));
        FusionRecipe.Constant constant =
                builder.addConstant(
                        "unusedConstant", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 4, 4));
        FusionRecipe.Input bfloat =
                builder.addInput("bfloat", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 4));
        FusionRecipe.Input single =
                builder.addInput("single", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 5));
        FusionRecipe.OutputPack packed = builder.outputPack("packed", half, bfloat, single);
        FusionRecipe.Output output = builder.addOutput("scores", packed);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(recipe.getName(), "mixed-output-pack");
        Assert.assertSame(recipe.getDimensions().get(0), rows);
        Assert.assertEquals(rows.getIndex(), 0);
        Assert.assertEquals(rows.getMaximumExtent(), 32);
        Assert.assertEquals(half.getInputIndex(), 0);
        Assert.assertEquals(bfloat.getInputIndex(), 1);
        Assert.assertEquals(single.getInputIndex(), 2);
        Assert.assertEquals(half.getIndex(), 0);
        Assert.assertEquals(constant.getIndex(), 1);
        Assert.assertEquals(bfloat.getIndex(), 2);
        Assert.assertEquals(single.getIndex(), 3);
        Assert.assertEquals(packed.getIndex(), 4);
        Assert.assertEquals(constant.getConstantIndex(), 0);
        Assert.assertEquals(packed.getSpec().getDataType(), DataType.FLOAT32);
        Assert.assertSame(packed.getSpec().getLeadingDimension(), rows);
        Assert.assertEquals(packed.getSpec().getInnerShape(), new long[] {12});
        Assert.assertEquals(packed.getSpec().getMaximumShape().getShape(), new long[] {32, 12});
        Assert.assertEquals(packed.getSources().size(), 3);
        Assert.assertSame(packed.getSources().get(0), half);
        Assert.assertSame(packed.getSources().get(1), bfloat);
        Assert.assertSame(packed.getSources().get(2), single);
        Assert.assertEquals(output.getIndex(), 0);
        Assert.assertSame(output.getValue(), packed);
        Assert.assertSame(recipe.getOutputs().get(0), output);

        long[] innerShape = packed.getSpec().getInnerShape();
        innerShape[0] = 1;
        Assert.assertEquals(packed.getSpec().getInnerShape(), new long[] {12});
        Assert.assertThrows(UnsupportedOperationException.class, () -> recipe.getInputs().clear());
        Assert.assertThrows(UnsupportedOperationException.class, () -> packed.getSources().clear());
    }

    @Test
    public void outputPackRejectsIncompatibleSources() {
        FusionRecipe.Builder builder = FusionRecipe.builder("invalid-output-pack");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 8);
        FusionRecipe.Dimension otherRows = builder.addDimension("otherRows", 8);
        FusionRecipe.Input valid =
                builder.addInput("valid", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2));
        FusionRecipe.Input otherExtent =
                builder.addInput(
                        "otherExtent", FusionRecipe.TensorSpec.of(DataType.FLOAT32, otherRows, 2));
        FusionRecipe.Input integral =
                builder.addInput("integral", FusionRecipe.TensorSpec.of(DataType.INT32, rows, 2));
        FusionRecipe.Input rankThree =
                builder.addInput(
                        "rankThree", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 2));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.outputPack("differentExtent", valid, otherExtent));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.outputPack("integralPack", valid, integral));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.outputPack("rankThreePack", rankThree));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.outputPack("emptyPack", new FusionRecipe.Value[0]));
    }

    @Test
    public void segmentedOutputPackPreservesTypeAndTrailingShape() {
        FusionRecipe.Builder builder = FusionRecipe.builder("segmented-output-pack");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 16);
        FusionRecipe.Input round =
                builder.addInput(
                        "round", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 1, 256));
        FusionRecipe.Input players =
                builder.addInput(
                        "players",
                        FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4, 29, 256));
        FusionRecipe.Input tiles =
                builder.addInput(
                        "tiles", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 34, 256));

        FusionRecipe.SegmentedOutputPack packed =
                builder.segmentedOutputPack("memory")
                        .addSource(round)
                        .addSourceSlice(players, 0, 1)
                        .addSource(tiles)
                        .addSourceSlice(players, 1, 24)
                        .addSourceSlice(players, 25, 4)
                        .build();
        builder.addOutput("memory", packed);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(packed.getSpec().getDataType(), DataType.FLOAT16);
        Assert.assertSame(packed.getSpec().getLeadingDimension(), batch);
        Assert.assertEquals(packed.getSpec().getInnerShape(), new long[] {151, 256});
        Assert.assertEquals(
                packed.getSpec().getMaximumShape().getShape(), new long[] {16, 151, 256});
        Assert.assertEquals(
                packed.getSources(), Arrays.asList(round, players, tiles, players, players));
        Assert.assertEquals(packed.getSourceTokenOffsets(), new long[] {0, 0, 0, 1, 25});
        Assert.assertEquals(packed.getSourceTokenCounts(), new long[] {1, 1, 34, 24, 4});
        Assert.assertSame(recipe.getOutputs().get(0).getValue(), packed);
        Assert.assertThrows(UnsupportedOperationException.class, () -> packed.getSources().clear());
    }

    @Test
    public void segmentedOutputPackConvertsMixedSourcesIntoConfiguredType() {
        FusionRecipe.Builder builder = FusionRecipe.builder("mixed-segmented-output-pack");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 16);
        FusionRecipe.Input half =
                builder.addInput(
                        "half", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 2, 256));
        FusionRecipe.Input single =
                builder.addInput(
                        "single", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 3, 256));

        FusionRecipe.SegmentedOutputPack packed =
                builder.segmentedOutputPack("memory")
                        .addSource(half)
                        .addSource(single)
                        .optOutputDataType(DataType.FLOAT32)
                        .build();
        builder.addOutput("memory", packed);
        builder.build();

        Assert.assertEquals(packed.getSpec().getDataType(), DataType.FLOAT32);
        Assert.assertEquals(packed.getSpec().getInnerShape(), new long[] {5, 256});
    }

    @Test
    public void segmentedOutputPackRejectsIncompatibleSources() {
        FusionRecipe.Builder builder = FusionRecipe.builder("invalid-segmented-output-pack");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 8);
        FusionRecipe.Dimension otherBatch = builder.addDimension("otherBatch", 8);
        FusionRecipe.Input valid =
                builder.addInput(
                        "valid", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 2, 4));
        FusionRecipe.Input wrongRows =
                builder.addInput(
                        "wrongRows",
                        FusionRecipe.TensorSpec.of(DataType.FLOAT16, otherBatch, 2, 4));
        FusionRecipe.Input integral =
                builder.addInput(
                        "integral", FusionRecipe.TensorSpec.of(DataType.INT32, batch, 2, 4));
        FusionRecipe.Input wrongWidth =
                builder.addInput(
                        "wrongWidth", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 2, 5));
        FusionRecipe.Input rankTwo =
                builder.addInput(
                        "rankTwo", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.segmentedOutputPack("wrongRowsPack", valid, wrongRows));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.segmentedOutputPack("integralPack", valid, integral));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.segmentedOutputPack("wrongWidthPack", valid, wrongWidth));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.segmentedOutputPack("rankTwoPack", rankTwo));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.segmentedOutputPack(
                                "emptyPack", new FusionRecipe.Value[0]));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.segmentedOutputPack("invalidOutputType")
                                .addSource(valid)
                                .optOutputDataType(DataType.INT32));
    }

    @Test
    public void binaryBranchBlendBuildsPresenceAwareGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("binary-branch-blend");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 32);
        FusionRecipe.Input baseline =
                builder.addInput(
                        "baseline", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 256));
        FusionRecipe.Input selected =
                builder.addInput(
                        "selected", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 256));
        FusionRecipe.Input selectedLogit =
                builder.addInput(
                        "selectedLogit", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 1));
        FusionRecipe.Input baselinePresence =
                builder.addInput(
                        "baselinePresence", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 1));
        FusionRecipe.Input selectedPresence =
                builder.addInput(
                        "selectedPresence", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));

        FusionRecipe.BinaryBranchBlend blend =
                builder.binaryBranchBlend(
                        "blend",
                        baseline,
                        selected,
                        selectedLogit,
                        baselinePresence,
                        selectedPresence);
        builder.addOutput("output", blend);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(blend.getSpec().getDataType(), DataType.FLOAT32);
        Assert.assertSame(blend.getSpec().getLeadingDimension(), rows);
        Assert.assertEquals(blend.getSpec().getInnerShape(), new long[] {256});
        Assert.assertSame(blend.getBaselineContext(), baseline);
        Assert.assertSame(blend.getSelectedContext(), selected);
        Assert.assertSame(blend.getSelectedLogit(), selectedLogit);
        Assert.assertSame(blend.getBaselinePresence(), baselinePresence);
        Assert.assertSame(blend.getSelectedPresence(), selectedPresence);
        Assert.assertSame(recipe.getOutputs().get(0).getValue(), blend);
    }

    @Test
    public void binaryBranchBlendRejectsIncompatibleInputs() {
        FusionRecipe.Builder builder = FusionRecipe.builder("invalid-binary-branch");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 8);
        FusionRecipe.Dimension otherRows = builder.addDimension("otherRows", 8);
        FusionRecipe.Input baseline =
                builder.addInput("baseline", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 3));
        FusionRecipe.Input selected =
                builder.addInput("selected", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 3));
        FusionRecipe.Input scalar =
                builder.addInput("scalar", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
        FusionRecipe.Input wrongType =
                builder.addInput("wrongType", FusionRecipe.TensorSpec.of(DataType.INT16, rows, 1));
        FusionRecipe.Input wrongWidth =
                builder.addInput(
                        "wrongWidth", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2));
        FusionRecipe.Input wrongRows =
                builder.addInput(
                        "wrongRows", FusionRecipe.TensorSpec.of(DataType.FLOAT32, otherRows, 1));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.binaryBranchBlend(
                                "wrongTypeBlend", baseline, selected, wrongType, scalar, scalar));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.binaryBranchBlend(
                                "wrongWidthBlend", baseline, selected, scalar, wrongWidth, scalar));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.binaryBranchBlend(
                                "wrongRowsBlend", baseline, selected, scalar, scalar, wrongRows));
    }

    @Test
    public void affineSumBuildsBroadcastGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("affine-sum");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 32);
        FusionRecipe.Input candidates =
                builder.addInput(
                        "candidates", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 4, 3));
        FusionRecipe.Input context =
                builder.addInput(
                        "context", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 1, 2));
        FusionRecipe.Constant table =
                builder.addConstant(
                        "table", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 1, 4, 2));
        FusionRecipe.Constant candidateWeight =
                builder.addConstant(
                        "candidateWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 3));
        FusionRecipe.Constant contextWeight =
                builder.addConstant(
                        "contextWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 2));
        FusionRecipe.Constant tableWeight =
                builder.addConstant(
                        "tableWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 2));
        FusionRecipe.Constant bias =
                builder.addConstant("bias", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5));
        FusionRecipe.AffineSum affine =
                builder.affineSum("hidden", 5)
                        .addTerm(candidates, candidateWeight)
                        .addTerm(context, contextWeight)
                        .addTerm(table, tableWeight)
                        .optBias(bias)
                        .optActivation(FusionRecipe.Activation.SILU)
                        .build();
        builder.addOutput("output", affine);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(affine.getSpec().getDataType(), DataType.FLOAT16);
        Assert.assertSame(affine.getSpec().getLeadingDimension(), rows);
        Assert.assertEquals(affine.getSpec().getInnerShape(), new long[] {4, 5});
        Assert.assertEquals(affine.getTerms().size(), 3);
        Assert.assertSame(affine.getTerms().get(0).getInput(), candidates);
        Assert.assertSame(affine.getTerms().get(0).getWeight(), candidateWeight);
        Assert.assertSame(affine.getTerms().get(1).getInput(), context);
        Assert.assertSame(affine.getTerms().get(2).getInput(), table);
        Assert.assertSame(affine.getBias(), bias);
        Assert.assertEquals(affine.getActivation(), FusionRecipe.Activation.SILU);
        Assert.assertSame(recipe.getValues().get(7), affine);
        Assert.assertThrows(UnsupportedOperationException.class, () -> affine.getTerms().clear());
    }

    @Test
    public void affineSumUsesWeightDataTypeForMixedSources() {
        FusionRecipe.Builder builder = FusionRecipe.builder("mixed-affine-sum");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 16);
        FusionRecipe.Input single =
                builder.addInput(
                        "single", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 3));
        FusionRecipe.Input half =
                builder.addInput("half", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 2, 1));
        FusionRecipe.Constant singleWeight =
                builder.addConstant(
                        "singleWeight", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 4, 3));
        FusionRecipe.Constant halfWeight =
                builder.addConstant(
                        "halfWeight", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 4, 1));
        FusionRecipe.Constant bias =
                builder.addConstant("bias", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 4));

        FusionRecipe.AffineSum affine =
                builder.affineSum("hidden", 4)
                        .addTerm(single, singleWeight)
                        .addTerm(half, halfWeight)
                        .optBias(bias)
                        .build();
        builder.addOutput("output", affine);
        builder.build();

        Assert.assertEquals(affine.getSpec().getDataType(), DataType.BFLOAT16);
        Assert.assertEquals(affine.getSpec().getInnerShape(), new long[] {2, 4});
        Assert.assertEquals(
                affine.getTerms().get(0).getInput().getSpec().getDataType(), DataType.FLOAT32);
        Assert.assertEquals(
                affine.getTerms().get(1).getInput().getSpec().getDataType(), DataType.FLOAT16);
    }

    @Test
    public void affineSumRejectsInvalidTerms() {
        FusionRecipe.Builder empty = FusionRecipe.builder("empty-affine");
        Assert.assertThrows(
                IllegalStateException.class, () -> empty.affineSum("hidden", 3).build());

        FusionRecipe.Builder mismatched = FusionRecipe.builder("mismatched-affine");
        FusionRecipe.Dimension rows = mismatched.addDimension("rows", 8);
        FusionRecipe.Input left =
                mismatched.addInput(
                        "left", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 3));
        FusionRecipe.Input right =
                mismatched.addInput(
                        "right", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 3, 4));
        FusionRecipe.Constant leftWeight =
                mismatched.addConstant(
                        "leftWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 5, 3));
        FusionRecipe.Constant rightWeight =
                mismatched.addConstant(
                        "rightWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 5, 4));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        mismatched
                                .affineSum("hidden", 5)
                                .addTerm(left, leftWeight)
                                .addTerm(right, rightWeight)
                                .build());

        FusionRecipe.Builder badWeight = FusionRecipe.builder("bad-weight-affine");
        FusionRecipe.Dimension samples = badWeight.addDimension("samples", 8);
        FusionRecipe.Input first =
                badWeight.addInput(
                        "first", FusionRecipe.TensorSpec.of(DataType.FLOAT32, samples, 3));
        FusionRecipe.Input second =
                badWeight.addInput(
                        "second", FusionRecipe.TensorSpec.of(DataType.FLOAT16, samples, 2));
        FusionRecipe.Constant firstWeight =
                badWeight.addConstant(
                        "firstWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 5, 3));
        FusionRecipe.Constant secondWeight =
                badWeight.addConstant(
                        "secondWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 2));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        badWeight
                                .affineSum("hidden", 5)
                                .addTerm(first, firstWeight)
                                .addTerm(second, secondWeight)
                                .build());

        FusionRecipe.Builder badSource = FusionRecipe.builder("bad-source-affine");
        FusionRecipe.Dimension badSourceRows = badSource.addDimension("rows", 8);
        FusionRecipe.Input integral =
                badSource.addInput(
                        "integral", FusionRecipe.TensorSpec.of(DataType.INT32, badSourceRows, 3));
        FusionRecipe.Constant projection =
                badSource.addConstant(
                        "projection", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 3));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> badSource.affineSum("hidden", 5).addTerm(integral, projection).build());
    }

    @Test
    public void indexedAffineBuildsMixedGatherProjection() {
        FusionRecipe.Builder builder = FusionRecipe.builder("indexed-affine");
        FusionRecipe.Dimension batches = builder.addDimension("batches", 4);
        FusionRecipe.Dimension destinations = builder.addDimension("destinations", 16);
        FusionRecipe.Dimension active = builder.addDimension("active", 8);
        FusionRecipe.Input indices =
                builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
        FusionRecipe.Input state =
                builder.addInput("state", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batches, 2));
        FusionRecipe.Input branch =
                builder.addInput(
                        "branch", FusionRecipe.TensorSpec.of(DataType.FLOAT16, destinations, 4));
        FusionRecipe.Constant hiddenWeight =
                builder.addConstant(
                        "hiddenWeight", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 5, 6));
        FusionRecipe.Constant hiddenBias =
                builder.addConstant(
                        "hiddenBias", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 5));
        FusionRecipe.Constant outputWeight =
                builder.addConstant(
                        "outputWeight", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 2, 5));
        FusionRecipe.Constant outputBias =
                builder.addConstant(
                        "outputBias", FusionRecipe.TensorSpec.fixed(DataType.BFLOAT16, 2));

        FusionRecipe.IndexedAffine indexed =
                builder.indexedAffine("scores", indices, destinations)
                        .addSource(state, 4)
                        .addSource(branch, 1)
                        .setHiddenWeight(hiddenWeight)
                        .optHiddenBias(hiddenBias)
                        .optActivation(FusionRecipe.Activation.SILU)
                        .setOutputWeight(outputWeight)
                        .optOutputBias(outputBias)
                        .build();
        builder.addOutput("output", indexed);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(recipe.getValues().get(indexed.getIndex()), indexed);
        Assert.assertEquals(indexed.getSpec().getDataType(), DataType.BFLOAT16);
        Assert.assertSame(indexed.getSpec().getLeadingDimension(), destinations);
        Assert.assertEquals(indexed.getSpec().getInnerShape(), new long[] {2});
        Assert.assertSame(indexed.getIndices(), indices);
        Assert.assertEquals(indexed.getSources().size(), 2);
        Assert.assertSame(indexed.getSources().get(0).getInput(), state);
        Assert.assertEquals(indexed.getSources().get(0).getIndexDivisor(), 4L);
        Assert.assertSame(indexed.getHiddenWeight(), hiddenWeight);
        Assert.assertSame(indexed.getHiddenBias(), hiddenBias);
        Assert.assertEquals(indexed.getActivation(), FusionRecipe.Activation.SILU);
        Assert.assertSame(indexed.getOutputWeight(), outputWeight);
        Assert.assertSame(indexed.getOutputBias(), outputBias);
    }

    @Test
    public void indexedAffineRejectsInvalidMetadata() {
        FusionRecipe.Builder builder = FusionRecipe.builder("bad-indexed-affine");
        FusionRecipe.Dimension rows = builder.addDimension("rows", 8);
        FusionRecipe.Dimension active = builder.addDimension("active", 4);
        FusionRecipe.Input indices =
                builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT64, active));
        FusionRecipe.Input source =
                builder.addInput("source", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 3));
        FusionRecipe.Constant hiddenWeight =
                builder.addConstant(
                        "hiddenWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 3));
        FusionRecipe.Constant outputWeight =
                builder.addConstant(
                        "outputWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 1, 5));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> builder.indexedAffine("zeroDivisor", indices, rows).addSource(source, 0));
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        builder.indexedAffine("missingWeights", indices, rows)
                                .addSource(source, 1)
                                .build());

        FusionRecipe.Builder badIndices = FusionRecipe.builder("bad-indices");
        FusionRecipe.Dimension badRows = badIndices.addDimension("rows", 8);
        FusionRecipe.Dimension badActive = badIndices.addDimension("active", 4);
        FusionRecipe.Input floatingIndices =
                badIndices.addInput(
                        "indices", FusionRecipe.TensorSpec.of(DataType.FLOAT32, badActive));
        FusionRecipe.Input badSource =
                badIndices.addInput(
                        "source", FusionRecipe.TensorSpec.of(DataType.FLOAT16, badRows, 3));
        FusionRecipe.Constant badHidden =
                badIndices.addConstant(
                        "hidden", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 5, 3));
        FusionRecipe.Constant badOutput =
                badIndices.addConstant(
                        "output", FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, 1, 5));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        badIndices
                                .indexedAffine("scores", floatingIndices, badRows)
                                .addSource(badSource, 1)
                                .setHiddenWeight(badHidden)
                                .setOutputWeight(badOutput)
                                .build());
    }

    @Test
    public void builderRejectsForeignHandlesAndMutationAfterBuild() {
        FusionRecipe.Builder first = FusionRecipe.builder("first");
        FusionRecipe.Dimension firstRows = first.addDimension("rows", 4);
        FusionRecipe.Input firstInput =
                first.addInput(
                        "scores", FusionRecipe.TensorSpec.of(DataType.FLOAT32, firstRows, 2));

        FusionRecipe.Builder second = FusionRecipe.builder("second");
        FusionRecipe.Dimension secondRows = second.addDimension("rows", 4);
        FusionRecipe.Input secondInput =
                second.addInput(
                        "scores", FusionRecipe.TensorSpec.of(DataType.FLOAT32, secondRows, 2));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> first.outputPack("foreign", firstInput, secondInput));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        first.addInput(
                                "foreignDimension",
                                FusionRecipe.TensorSpec.of(DataType.FLOAT32, secondRows, 2)));

        FusionRecipe.OutputPack packed = first.outputPack("packed", firstInput);
        first.addOutput("output", packed);
        first.build();
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        first.addInput(
                                "late",
                                FusionRecipe.TensorSpec.of(DataType.FLOAT32, firstRows, 1)));
        Assert.assertThrows(IllegalStateException.class, first::build);
    }

    @Test
    public void transformerEncoderStackBuildsFixedSequenceGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("short-transformer");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Input input =
                builder.addInput(
                        "tokens", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 6, 256));
        FusionRecipe.Constant attentionInputWeight = vector(builder, "attnNormWeight", 256, true);
        FusionRecipe.Constant attentionInputBias = vector(builder, "attnNormBias", 256, true);
        FusionRecipe.Constant queryKeyValue = matrix(builder, "qkv", 384, 256);
        FusionRecipe.Constant attentionOutput = matrix(builder, "attentionOutput", 256, 128);
        FusionRecipe.Constant attentionOutputBias =
                vector(builder, "attentionOutputBias", 256, false);
        FusionRecipe.Constant feedForwardInputWeight = vector(builder, "ffNormWeight", 256, true);
        FusionRecipe.Constant feedForwardInputBias = vector(builder, "ffNormBias", 256, true);
        FusionRecipe.Constant expansion = matrix(builder, "expansion", 512, 256);
        FusionRecipe.Constant expansionBias = vector(builder, "expansionBias", 512, false);
        FusionRecipe.Constant projection = matrix(builder, "projection", 256, 512);
        FusionRecipe.Constant projectionBias = vector(builder, "projectionBias", 256, false);
        FusionRecipe.Constant outputWeight = vector(builder, "outputNormWeight", 256, true);
        FusionRecipe.Constant outputBias = vector(builder, "outputNormBias", 256, true);

        FusionRecipe.TransformerEncoderStack stack =
                builder.transformerEncoderStack("stack", input, 4, 128, 512)
                        .addBlock(
                                attentionInputWeight,
                                attentionInputBias,
                                queryKeyValue,
                                attentionOutput,
                                attentionOutputBias,
                                feedForwardInputWeight,
                                feedForwardInputBias,
                                expansion,
                                expansionBias,
                                projection,
                                projectionBias,
                                outputWeight,
                                outputBias)
                        .build();
        builder.addOutput("encoded", stack);
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(stack.getSpec().getMaximumShape().getShape(), new long[] {384, 6, 256});
        Assert.assertEquals(stack.getBlocks().size(), 1);
        Assert.assertSame(stack.getBlocks().get(0).getQueryKeyValueWeight(), queryKeyValue);
        Assert.assertEquals(stack.getAttentionHeads(), 4);
        Assert.assertEquals(stack.getAttentionWidth(), 128);
        Assert.assertEquals(stack.getFeedForwardWidth(), 512);
        Assert.assertEquals(recipe.getOutputs().get(0).getValue(), stack);
    }

    @Test
    public void indexedLocalTransformerBuildsIndependentExtentGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("indexed-local-transformer");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Dimension active = builder.addDimension("active", 24_576);
        FusionRecipe.Input input =
                builder.addInput(
                        "tokens", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4, 29, 256));
        FusionRecipe.Input indices =
                builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
        FusionRecipe.Constant inputNormWeight = vector(builder, "inputNormWeight", 256, true);
        FusionRecipe.Constant inputNormBias = vector(builder, "inputNormBias", 256, true);
        FusionRecipe.Constant attentionInputWeight = vector(builder, "attnNormWeight", 256, true);
        FusionRecipe.Constant attentionInputBias = vector(builder, "attnNormBias", 256, true);
        FusionRecipe.Constant queryKeyValue = matrix(builder, "qkv", 192, 256);
        FusionRecipe.Constant attentionOutput = matrix(builder, "attentionOutput", 256, 64);
        FusionRecipe.Constant attentionOutputBias =
                vector(builder, "attentionOutputBias", 256, false);
        FusionRecipe.Constant feedForwardInputWeight = vector(builder, "ffNormWeight", 256, true);
        FusionRecipe.Constant feedForwardInputBias = vector(builder, "ffNormBias", 256, true);
        FusionRecipe.Constant expansion = matrix(builder, "expansion", 128, 256);
        FusionRecipe.Constant expansionBias = vector(builder, "expansionBias", 128, false);
        FusionRecipe.Constant projection = matrix(builder, "projection", 256, 128);
        FusionRecipe.Constant projectionBias = vector(builder, "projectionBias", 256, false);
        FusionRecipe.Constant outputWeight = vector(builder, "outputNormWeight", 256, true);
        FusionRecipe.Constant outputBias = vector(builder, "outputNormBias", 256, true);

        FusionRecipe.IndexedLocalTransformerEncoder encoder =
                builder.indexedLocalTransformerEncoder("encoded", input, indices, 4, 64, 128)
                        .setInputNormalization(inputNormWeight, inputNormBias)
                        .setBlock(
                                attentionInputWeight,
                                attentionInputBias,
                                queryKeyValue,
                                attentionOutput,
                                attentionOutputBias,
                                feedForwardInputWeight,
                                feedForwardInputBias,
                                expansion,
                                expansionBias,
                                projection,
                                projectionBias,
                                outputWeight,
                                outputBias)
                        .build();
        builder.addOutput("output", encoder);
        FusionRecipe recipe = builder.build();

        Assert.assertSame(encoder.getInput(), input);
        Assert.assertSame(encoder.getIndices(), indices);
        Assert.assertSame(encoder.getInputNormWeight(), inputNormWeight);
        Assert.assertSame(encoder.getBlock().getQueryKeyValueWeight(), queryKeyValue);
        Assert.assertEquals(encoder.getAttentionHeads(), 4);
        Assert.assertEquals(encoder.getAttentionWidth(), 64);
        Assert.assertEquals(encoder.getFeedForwardWidth(), 128);
        Assert.assertEquals(
                encoder.getSpec().getMaximumShape().getShape(), new long[] {384, 4, 29, 256});
        Assert.assertSame(recipe.getOutputs().get(0).getValue(), encoder);
    }

    @Test
    public void indexedLocalTransformerBuildsLogicalInputSegments() {
        FusionRecipe.Builder builder = FusionRecipe.builder("segmented-indexed-local-transformer");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Dimension active = builder.addDimension("active", 24_576);
        FusionRecipe.Input player =
                builder.addInput(
                        "player", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4, 1, 256));
        FusionRecipe.Input river =
                builder.addInput(
                        "river", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4, 24, 256));
        FusionRecipe.Input meld =
                builder.addInput(
                        "meld", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 4, 4, 256));
        FusionRecipe.Input indices =
                builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
        FusionRecipe.Constant inputNormWeight = vector(builder, "inputNormWeight", 256, true);
        FusionRecipe.Constant inputNormBias = vector(builder, "inputNormBias", 256, true);
        FusionRecipe.Constant attentionInputWeight = vector(builder, "attnNormWeight", 256, true);
        FusionRecipe.Constant attentionInputBias = vector(builder, "attnNormBias", 256, true);
        FusionRecipe.Constant queryKeyValue = matrix(builder, "qkv", 192, 256);
        FusionRecipe.Constant attentionOutput = matrix(builder, "attentionOutput", 256, 64);
        FusionRecipe.Constant attentionOutputBias =
                vector(builder, "attentionOutputBias", 256, false);
        FusionRecipe.Constant feedForwardInputWeight = vector(builder, "ffNormWeight", 256, true);
        FusionRecipe.Constant feedForwardInputBias = vector(builder, "ffNormBias", 256, true);
        FusionRecipe.Constant expansion = matrix(builder, "expansion", 128, 256);
        FusionRecipe.Constant expansionBias = vector(builder, "expansionBias", 128, false);
        FusionRecipe.Constant projection = matrix(builder, "projection", 256, 128);
        FusionRecipe.Constant projectionBias = vector(builder, "projectionBias", 256, false);
        FusionRecipe.Constant outputWeight = vector(builder, "outputNormWeight", 256, true);
        FusionRecipe.Constant outputBias = vector(builder, "outputNormBias", 256, true);

        FusionRecipe.IndexedLocalTransformerEncoder encoder =
                builder.indexedLocalTransformerEncoder(
                                "encoded", Arrays.asList(player, river, meld), indices, 4, 64, 128)
                        .setInputNormalization(inputNormWeight, inputNormBias)
                        .setBlock(
                                attentionInputWeight,
                                attentionInputBias,
                                queryKeyValue,
                                attentionOutput,
                                attentionOutputBias,
                                feedForwardInputWeight,
                                feedForwardInputBias,
                                expansion,
                                expansionBias,
                                projection,
                                projectionBias,
                                outputWeight,
                                outputBias)
                        .build();
        builder.addOutput("output", encoder);
        builder.build();

        Assert.assertEquals(encoder.getInputSegments(), Arrays.asList(player, river, meld));
        Assert.assertThrows(IllegalStateException.class, encoder::getInput);
        Assert.assertEquals(
                encoder.getSpec().getMaximumShape().getShape(), new long[] {384, 4, 29, 256});
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> encoder.getInputSegments().clear());
    }

    @Test
    public void transformerEncoderStackBuildsIndexedRelationGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("relation-transformer");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Input input =
                builder.addInput(
                        "tokens", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 34, 256));
        FusionRecipe.Constant relationIds =
                builder.addConstant(
                        "relationIds", FusionRecipe.TensorSpec.fixed(DataType.INT16, 34, 34));
        FusionRecipe.Constant relationKeys = matrix(builder, "relationKeys", 23, 64);
        FusionRecipe.Constant relationBias = matrix(builder, "relationBias", 23, 4);
        IndexedRelationAttention relation =
                builder.indexedRelationAttention(relationIds, relationKeys, relationBias);
        FusionRecipe.TransformerEncoderStack stack =
                builder.transformerEncoderStack("stack", input, 4, 64, 128)
                        .addBlock(
                                vector(builder, "attnNormWeight", 256, true),
                                vector(builder, "attnNormBias", 256, true),
                                matrix(builder, "qkv", 192, 256),
                                matrix(builder, "attentionOutput", 256, 64),
                                vector(builder, "attentionOutputBias", 256, false),
                                vector(builder, "ffNormWeight", 256, true),
                                vector(builder, "ffNormBias", 256, true),
                                matrix(builder, "expansion", 128, 256),
                                vector(builder, "expansionBias", 128, false),
                                matrix(builder, "projection", 256, 128),
                                vector(builder, "projectionBias", 256, false),
                                vector(builder, "outputNormWeight", 256, true),
                                vector(builder, "outputNormBias", 256, true),
                                relation)
                        .build();

        Assert.assertTrue(stack.hasIndexedRelationAttention());
        Assert.assertSame(
                stack.getBlocks().get(0).getIndexedRelationAttention().getRelationIds(),
                relationIds);
        Assert.assertSame(
                stack.getBlocks().get(0).getIndexedRelationAttention().getRelationKeys(),
                relationKeys);
        Assert.assertSame(
                stack.getBlocks().get(0).getIndexedRelationAttention().getRelationBias(),
                relationBias);
    }

    @Test
    public void transformerEncoderStackRejectsNonInt16RelationIds() {
        FusionRecipe.Builder builder = FusionRecipe.builder("invalid-relation-transformer");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 2);
        FusionRecipe.Input input =
                builder.addInput(
                        "tokens", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 34, 256));
        IndexedRelationAttention relation =
                builder.indexedRelationAttention(
                        builder.addConstant(
                                "relationIds",
                                FusionRecipe.TensorSpec.fixed(DataType.INT32, 34, 34)),
                        matrix(builder, "relationKeys", 23, 64),
                        matrix(builder, "relationBias", 23, 4));
        FusionRecipe.TransformerEncoderStackBuilder stack =
                builder.transformerEncoderStack("stack", input, 4, 64, 128)
                        .addBlock(
                                vector(builder, "attnNormWeight", 256, true),
                                vector(builder, "attnNormBias", 256, true),
                                matrix(builder, "qkv", 192, 256),
                                matrix(builder, "attentionOutput", 256, 64),
                                vector(builder, "attentionOutputBias", 256, false),
                                vector(builder, "ffNormWeight", 256, true),
                                vector(builder, "ffNormBias", 256, true),
                                matrix(builder, "expansion", 128, 256),
                                vector(builder, "expansionBias", 128, false),
                                matrix(builder, "projection", 256, 128),
                                vector(builder, "projectionBias", 256, false),
                                vector(builder, "outputNormWeight", 256, true),
                                vector(builder, "outputNormBias", 256, true),
                                relation);

        Assert.assertThrows(IllegalArgumentException.class, stack::build);
    }

    @Test
    public void singleQueryReadoutGroupBuildsSharedMemoryGraph() {
        FusionRecipe.Builder builder = FusionRecipe.builder("single-query-readouts");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Input memory =
                builder.addInput(
                        "memory", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 151, 256));
        FusionRecipe.Input querySource =
                builder.addInput(
                        "querySource", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 6, 256));
        FusionRecipe.Input mask =
                builder.addInput("mask", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 151));
        FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder groupBuilder =
                builder.singleQueryCrossAttentionReadoutGroup(
                                "readouts", memory, querySource, mask, 4)
                        .optQueryIndex(3);
        addReadout(builder, groupBuilder, "policy", 384);
        addReadout(builder, groupBuilder, "value", 256);
        FusionRecipe.SingleQueryCrossAttentionReadoutGroup group = groupBuilder.build();
        builder.addOutput("policy", group.getReadoutState(0));
        builder.addOutput("value", group.getReadoutState(1));
        FusionRecipe recipe = builder.build();

        Assert.assertEquals(
                group.getReadoutState(0).getSpec().getMaximumShape().getShape(),
                new long[] {384, 256});
        Assert.assertEquals(
                group.getReadoutState(1).getSpec().getMaximumShape().getShape(),
                new long[] {384, 256});
        Assert.assertEquals(group.getReadoutState(0).getSpec().getDataType(), DataType.FLOAT16);
        Assert.assertSame(group.getMemory(), memory);
        Assert.assertSame(group.getQuerySource(), querySource);
        Assert.assertEquals(group.getQueryIndex(), 3);
        Assert.assertSame(group.getValidMask(), mask);
        Assert.assertEquals(group.getAttentionHeads(), 4);
        Assert.assertEquals(group.getAttentionWidth(), 64);
        Assert.assertEquals(group.getMaximumFeedForwardWidth(), 384);
        Assert.assertEquals(group.getReadouts().size(), 2);
        Assert.assertEquals(group.getReadouts().get(0).getFeedForwardWidth(), 384);
        Assert.assertEquals(group.getReadouts().get(1).getFeedForwardWidth(), 256);
        Assert.assertSame(recipe.getOutputs().get(0).getValue(), group.getReadoutState(0));
        Assert.assertSame(recipe.getOutputs().get(1).getValue(), group.getReadoutState(1));
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> group.getReadoutStates().clear());
        Assert.assertThrows(UnsupportedOperationException.class, () -> group.getReadouts().clear());
    }

    @Test
    public void mappedGroupedMaskedSoftmaxPoolBuildsContiguousOutputSets() {
        FusionRecipe.Builder builder = FusionRecipe.builder("mapped-candidate-pool");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 384);
        FusionRecipe.Input scores =
                builder.addInput("scores", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 16));
        FusionRecipe.Input masks =
                builder.addInput(
                        "masks", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, batch, 16, 11));
        FusionRecipe.Input values =
                builder.addInput(
                        "values", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 16, 256));
        FusionRecipe.Constant alternatives =
                builder.addConstant(
                        "alternatives", FusionRecipe.TensorSpec.fixed(DataType.INT32, 10));
        FusionRecipe.Constant pass =
                builder.addConstant("pass", FusionRecipe.TensorSpec.fixed(DataType.INT64, 1));
        FusionRecipe.MappedGroupedMaskedSoftmaxPoolGroup group =
                builder.mappedGroupedMaskedSoftmaxPoolGroup("candidateTypes", scores, masks, values)
                        .addOutputSet("alternatives", alternatives)
                        .addOutputSet("pass", pass)
                        .build();
        builder.addOutput("alternativeContexts", group.getOutputSet(0).getContexts());
        builder.addOutput("alternativePresence", group.getOutputSet(0).getPresence());
        builder.addOutput("passContext", group.getOutputSet(1).getContexts());
        builder.addOutput("passPresence", group.getOutputSet(1).getPresence());
        FusionRecipe recipe = builder.build();

        Assert.assertSame(group.getScores(), scores);
        Assert.assertSame(group.getMasks(), masks);
        Assert.assertSame(group.getValues(), values);
        Assert.assertEquals(group.getOutputSets().size(), 2);
        Assert.assertSame(group.getOutputSet(0).getDestinationGroupIndices(), alternatives);
        Assert.assertSame(group.getOutputSet(1).getDestinationGroupIndices(), pass);
        Assert.assertEquals(
                group.getOutputSet(0).getContexts().getSpec().getMaximumShape().getShape(),
                new long[] {384, 10, 256});
        Assert.assertEquals(
                group.getOutputSet(0).getContexts().getSpec().getDataType(), DataType.FLOAT32);
        Assert.assertEquals(
                group.getOutputSet(0).getPresence().getSpec().getMaximumShape().getShape(),
                new long[] {384, 10});
        Assert.assertEquals(
                group.getOutputSet(0).getPresence().getSpec().getDataType(), DataType.BFLOAT16);
        Assert.assertEquals(
                group.getOutputSet(1).getContexts().getSpec().getMaximumShape().getShape(),
                new long[] {384, 1, 256});
        Assert.assertEquals(
                group.getOutputSet(1).getPresence().getSpec().getMaximumShape().getShape(),
                new long[] {384, 1});
        Assert.assertEquals(recipe.getOutputs().size(), 4);
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> group.getOutputSets().clear());
    }

    @Test
    public void mappedGroupedMaskedSoftmaxPoolRejectsInvalidSchema() {
        FusionRecipe.Builder builder = FusionRecipe.builder("invalid-mapped-pool");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 8);
        FusionRecipe.Input scores =
                builder.addInput("scores", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 4));
        FusionRecipe.Input masks =
                builder.addInput(
                        "masks", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 4, 3));
        FusionRecipe.Input values =
                builder.addInput(
                        "values", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 4, 6));
        FusionRecipe.Constant mapping =
                builder.addConstant("mapping", FusionRecipe.TensorSpec.fixed(DataType.INT32, 3));
        FusionRecipe.Constant floatingMapping =
                builder.addConstant(
                        "floatingMapping", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 3));

        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        builder.mappedGroupedMaskedSoftmaxPoolGroup(
                                        "badMapping", scores, masks, values)
                                .addOutputSet("set", floatingMapping));
        FusionRecipe.MappedGroupedMaskedSoftmaxPoolGroupBuilder groupBuilder =
                builder.mappedGroupedMaskedSoftmaxPoolGroup("pool", scores, masks, values)
                        .addOutputSet("set", mapping);
        Assert.assertThrows(
                IllegalArgumentException.class, () -> groupBuilder.addOutputSet("set", mapping));
    }

    private static void addReadout(
            FusionRecipe.Builder builder,
            FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder group,
            String prefix,
            int feedForwardWidth) {
        FusionRecipe.Constant seedWeight = matrix(builder, prefix + "SeedWeight", 256, 512);
        FusionRecipe.Constant seedBias = vector(builder, prefix + "SeedBias", 256, false);
        FusionRecipe.Constant queryWeight = matrix(builder, prefix + "QueryWeight", 64, 256);
        FusionRecipe.Constant queryBias = vector(builder, prefix + "QueryBias", 64, false);
        FusionRecipe.Constant keyValue = matrix(builder, prefix + "KeyValue", 128, 256);
        FusionRecipe.Constant contextWeight = matrix(builder, prefix + "ContextWeight", 256, 64);
        FusionRecipe.Constant contextBias = vector(builder, prefix + "ContextBias", 256, false);
        FusionRecipe.Constant queryNormWeight =
                vector(builder, prefix + "QueryNormWeight", 256, true);
        FusionRecipe.Constant queryNormBias = vector(builder, prefix + "QueryNormBias", 256, true);
        FusionRecipe.Constant feedForwardNormWeight =
                vector(builder, prefix + "FeedForwardNormWeight", 256, true);
        FusionRecipe.Constant feedForwardNormBias =
                vector(builder, prefix + "FeedForwardNormBias", 256, true);
        FusionRecipe.Constant expansion =
                matrix(builder, prefix + "Expansion", feedForwardWidth, 256);
        FusionRecipe.Constant expansionBias =
                vector(builder, prefix + "ExpansionBias", feedForwardWidth, false);
        FusionRecipe.Constant projection =
                matrix(builder, prefix + "Projection", 256, feedForwardWidth);
        FusionRecipe.Constant projectionBias =
                vector(builder, prefix + "ProjectionBias", 256, false);
        FusionRecipe.Constant outputNormWeight =
                vector(builder, prefix + "OutputNormWeight", 256, true);
        FusionRecipe.Constant outputNormBias =
                vector(builder, prefix + "OutputNormBias", 256, true);
        group.addReadout(
                seedWeight,
                seedBias,
                queryWeight,
                queryBias,
                keyValue,
                contextWeight,
                contextBias,
                queryNormWeight,
                queryNormBias,
                feedForwardNormWeight,
                feedForwardNormBias,
                expansion,
                expansionBias,
                projection,
                projectionBias,
                outputNormWeight,
                outputNormBias);
    }

    private static FusionRecipe.Constant vector(
            FusionRecipe.Builder builder, String name, int width, boolean norm) {
        return builder.addConstant(
                name,
                FusionRecipe.TensorSpec.fixed(norm ? DataType.FLOAT32 : DataType.FLOAT16, width));
    }

    private static FusionRecipe.Constant matrix(
            FusionRecipe.Builder builder, String name, int rows, int columns) {
        return builder.addConstant(
                name, FusionRecipe.TensorSpec.fixed(DataType.FLOAT16, rows, columns));
    }
}
