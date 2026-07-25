"use client";

/**
 * Card-attrition demo page.
 *
 * The API attaches a caveat to every score here — n = 10,127 and the dataset is
 * easy — and this page renders it rather than dropping it. A high number on this
 * problem is a property of the data, not evidence of a strong model.
 */

import { zodResolver } from "@hookform/resolvers/zod";

import { ModelPage, type ResultView } from "@/components/ModelPage";
import type { Tone } from "@/components/PredictionResult";
import {
  explainChurn,
  predictChurn,
  type ChurnPrediction,
  type RetentionAction,
} from "@/lib/api";
import { ChurnInputSchema, CHURN_DEFAULTS, type ChurnInput } from "@/lib/schemas";
import { CHURN_FIELDS } from "./fields";

const ACTION_TONE: Record<RetentionAction, Tone> = {
  NO_ACTION: "good",
  PROACTIVE_CHECKIN: "warn",
  URGENT_OUTREACH: "bad",
};

function toResultView(prediction: ChurnPrediction): ResultView {
  return {
    decisionLabel: "Retention action",
    decision: prediction.retention_action,
    decisionTone: ACTION_TONE[prediction.retention_action],
    probabilityLabel: "Attrition probability",
    probability: prediction.attrition_probability,
    modelVersion: prediction.model_version,
    trainedOn: prediction.trained_on,
    caveat: prediction.caveat,
  };
}

export default function ChurnPage() {
  return (
    <ModelPage<ChurnInput, ChurnPrediction>
      title="Card Attrition"
      intro="Score a cardholder against a LightGBM model trained on 10,127 real bank customers, evaluated by 5-fold stratified cross-validation."
      resolver={zodResolver(ChurnInputSchema)}
      defaults={CHURN_DEFAULTS}
      fields={CHURN_FIELDS}
      predict={predictChurn}
      explain={explainChurn}
      toResultView={toResultView}
    />
  );
}
