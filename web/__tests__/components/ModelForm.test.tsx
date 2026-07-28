/** A.2.5: ModelForm component tests. */
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

import { ModelForm, type FieldConfig } from "@/components/ModelForm";

// A small test schema with three representative field types.
const TestSchema = z.object({
  age: z.number().int().min(18).max(100),
  name: z.string().min(1),
  rate: z.number().min(0).max(1),
});
type TestInput = z.infer<typeof TestSchema>;

const FIELDS: FieldConfig<TestInput>[] = [
  { name: "age", label: "Age", type: "number", min: 18, max: 100 },
  { name: "name", label: "Name", type: "text" },
  { name: "rate", label: "Rate", type: "slider", min: 0, max: 1, step: 0.01 },
];

// Wrapper that wires react-hook-form for the component under test.
function Harness({
  onSubmit,
  defaults,
}: {
  onSubmit: (v: TestInput) => void;
  defaults?: Partial<TestInput>;
}) {
  const form = useForm<TestInput>({
    resolver: zodResolver(TestSchema),
    defaultValues: {
      age: defaults?.age ?? 25,
      name: defaults?.name ?? "Alice",
      rate: defaults?.rate ?? 0.5,
    },
  });
  return <ModelForm form={form} fields={FIELDS} onSubmit={onSubmit} />;
}

describe("ModelForm", () => {
  it("renders a labeled control for each field plus a submit button", () => {
    render(<Harness onSubmit={vi.fn()} />);
    // Input-backed fields expose their native labels
    expect(screen.getByLabelText("Age")).toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toBeInTheDocument();
    // Slider is a role="slider" element, with label text rendered separately
    expect(screen.getByText("Rate")).toBeInTheDocument();
    expect(screen.getByRole("slider")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit/i })).toBeInTheDocument();
  });

  it("calls onSubmit with typed values when defaults are valid", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<Harness onSubmit={onSubmit} defaults={{ age: 30, name: "Bob", rate: 0.75 }} />);

    await user.click(screen.getByRole("button", { name: /submit/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    const received = onSubmit.mock.calls[0][0];
    expect(received).toEqual({ age: 30, name: "Bob", rate: 0.75 });
    expect(typeof received.age).toBe("number");
    expect(typeof received.rate).toBe("number");
  });

  it("blocks onSubmit when input is invalid (Zod rejection)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    // Invalid defaults: age below the min(18) bound, name empty
    render(<Harness onSubmit={onSubmit} defaults={{ age: 15, name: "", rate: 0.5 }} />);

    await user.click(screen.getByRole("button", { name: /submit/i }));

    // Give RHF + Zod time to run and settle
    await new Promise((r) => setTimeout(r, 100));

    // Primary contract: invalid submissions must not reach the consumer's
    // onSubmit handler. That is the guarantee ModelForm needs to provide.
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("passes the edited numeric value to onSubmit as a number", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<Harness onSubmit={onSubmit} defaults={{ age: 25, name: "Alice", rate: 0.5 }} />);

    // userEvent.type on type=number inputs is flaky in jsdom; fireEvent.change
    // is the idiomatic workaround and triggers the same react-hook-form path.
    const ageInput = screen.getByLabelText("Age") as HTMLInputElement;
    fireEvent.change(ageInput, { target: { value: "55" } });

    await user.click(screen.getByRole("button", { name: /submit/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    const received = onSubmit.mock.calls[0][0];
    expect(received.age).toBe(55);
    expect(typeof received.age).toBe("number");
  });

  it("renders a select control with the provided options (added in A.9.4)", async () => {
    const SelectSchema = z.object({ country: z.string() });
    type SelectInput = z.infer<typeof SelectSchema>;
    const fields: FieldConfig<SelectInput>[] = [
      { name: "country", label: "Country", type: "select", options: ["united_states", "canada"] },
    ];
    const onSubmit = vi.fn();

    function SelectHarness() {
      const form = useForm<SelectInput>({
        resolver: zodResolver(SelectSchema),
        defaultValues: { country: "united_states" },
      });
      return <ModelForm form={form} fields={fields} onSubmit={onSubmit} />;
    }

    render(<SelectHarness />);
    const select = screen.getByLabelText(/country/i) as HTMLSelectElement;
    // Underscores in option values are humanized for display.
    expect(select).toBeInTheDocument();
    expect(select.value).toBe("united_states");
    expect(screen.getByRole("option", { name: "United States" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Canada" })).toBeInTheDocument();

    // Submitting passes the raw value (snake_case), not the humanized label.
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /submit/i }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(onSubmit.mock.calls[0][0].country).toBe("united_states");
  });
});
