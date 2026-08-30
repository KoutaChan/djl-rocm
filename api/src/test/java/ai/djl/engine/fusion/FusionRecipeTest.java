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
}
