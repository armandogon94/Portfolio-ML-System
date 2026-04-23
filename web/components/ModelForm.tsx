"use client";

/**
 * ModelForm — reusable wrapper for all industry model-input forms.
 *
 * Renders each field declared in `fields` using shadcn/ui Form primitives,
 * with inline Zod validation errors and a submit button at the bottom.
 * Supports three field types: number (<Input type="number">), text
 * (<Input type="text">), and slider (shadcn <Slider>). `select` is
 * reserved for future work and throws until implemented, keeping the
 * type surface closed.
 *
 * Composition pattern (from tasks/plan.md §A.2.5 REFACTOR step):
 * ModelForm iterates fields and delegates each to <ModelFormField/>,
 * keeping the outer component readable as the field types grow.
 */

import * as React from "react";
import type {
  UseFormReturn,
  FieldValues,
  Path,
  PathValue,
  ControllerRenderProps,
} from "react-hook-form";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";

export type FieldConfig<TValues extends FieldValues> = {
  name: Path<TValues>;
  label: string;
  type: "number" | "text" | "slider" | "select";
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description?: string;
};

type ModelFormProps<TValues extends FieldValues> = {
  form: UseFormReturn<TValues>;
  fields: FieldConfig<TValues>[];
  onSubmit: (values: TValues) => void | Promise<void>;
  submitLabel?: string;
  isSubmitting?: boolean;
};

export function ModelForm<TValues extends FieldValues>({
  form,
  fields,
  onSubmit,
  submitLabel = "Submit",
  isSubmitting,
}: ModelFormProps<TValues>) {
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        {fields.map((field) => (
          <ModelFormField<TValues> key={String(field.name)} form={form} field={field} />
        ))}
        <Button type="submit" disabled={isSubmitting ?? form.formState.isSubmitting}>
          {submitLabel}
        </Button>
      </form>
    </Form>
  );
}

// Private: per-field renderer. Kept out of ModelForm's body so the
// outer map stays small.
function ModelFormField<TValues extends FieldValues>({
  form,
  field,
}: {
  form: UseFormReturn<TValues>;
  field: FieldConfig<TValues>;
}) {
  return (
    <FormField
      control={form.control}
      name={field.name}
      render={({ field: rhf }) => (
        <FormItem>
          <FormLabel>{field.label}</FormLabel>
          <FormControl>{renderControl(field, rhf)}</FormControl>
          {field.description && (
            <p className="text-sm text-muted-foreground">{field.description}</p>
          )}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

function renderControl<TValues extends FieldValues>(
  field: FieldConfig<TValues>,
  rhf: ControllerRenderProps<TValues, Path<TValues>>,
) {
  switch (field.type) {
    case "number":
      // Coerce the raw input string to a number so the value handed to
      // Zod (and onSubmit) is typed correctly. An empty input becomes
      // undefined so Zod can flag it as a required field rather than
      // accepting NaN silently.
      return (
        <Input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step}
          value={rhf.value ?? ""}
          onChange={(e) =>
            rhf.onChange(
              e.target.value === ""
                ? undefined
                : (e.target.valueAsNumber as PathValue<TValues, Path<TValues>>),
            )
          }
          onBlur={rhf.onBlur}
          name={rhf.name}
          ref={rhf.ref}
        />
      );
    case "text":
      return <Input type="text" {...rhf} value={rhf.value ?? ""} />;
    case "slider":
      return (
        <Slider
          min={field.min ?? 0}
          max={field.max ?? 1}
          step={field.step ?? 0.01}
          value={[Number(rhf.value ?? field.min ?? 0)]}
          onValueChange={([v]) =>
            rhf.onChange(v as PathValue<TValues, Path<TValues>>)
          }
        />
      );
    case "select":
      throw new Error(
        "ModelForm: 'select' field type not implemented yet. " +
          "Use 'text' with a datalist, or extend renderControl().",
      );
  }
}
