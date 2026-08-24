import type { PaymentDetails, PaymentSummary } from "./api";

export const demoPayments: PaymentSummary[] = [
  { id: "pay_RV92K7M", amount: 2480, bank: "HDFC Bank", errorCode: "GATEWAY_TIMEOUT", rootCause: "BANK_DOWNTIME", action: "SMART_RETRY", status: "RECOVERED", timestamp: "2026-08-23T10:14:00+05:30" },
  { id: "pay_RV92K81", amount: 899, bank: "Kotak Mahindra", errorCode: "DECLINED", rootCause: "INSUFFICIENT_FUNDS", action: "WHATSAPP_LINK", status: "IN_FLIGHT", timestamp: "2026-08-23T10:11:00+05:30" },
  { id: "pay_RV92K44", amount: 12650, bank: "ICICI Bank", errorCode: "UPI_COLLECT_EXPIRED", rootCause: "USER_ABANDONMENT", action: "SMART_RETRY", status: "QUEUED", timestamp: "2026-08-23T10:07:00+05:30" },
  { id: "pay_RV92J97", amount: 1599, bank: "State Bank of India", errorCode: "ISSUER_DECLINED", rootCause: "BANK_DOWNTIME", action: "RETRY_LATER", status: "RECOVERED", timestamp: "2026-08-23T10:01:00+05:30" },
  { id: "pay_RV92J53", amount: 3499, bank: "Axis Bank", errorCode: "INSUFFICIENT_BALANCE", rootCause: "INSUFFICIENT_FUNDS", action: "EMAIL_REMINDER", status: "FAILED", timestamp: "2026-08-23T09:56:00+05:30" },
];

export const demoPaymentDetails: PaymentDetails = {
  id: "pay_RV92K7M",
  amount: 2480,
  currency: "INR",
  bank: "HDFC Bank",
  timestamp: "2026-08-23T10:14:00+05:30",
  initialErrorCode: "GATEWAY_TIMEOUT",
  userId: "usr_9D02A71",
  rootCause: "BANK_DOWNTIME",
  confidence: 0.92,
  shapExplanations: [
    { feature: "issuer_error_rate_10m", value: "18.4%", shap_value: 0.46, direction: "pushed toward BANK_DOWNTIME" },
    { feature: "bank_latency_p95", value: "11.8 sec", shap_value: 0.31, direction: "pushed toward BANK_DOWNTIME" },
    { feature: "user_balance", value: "₹18,300", shap_value: -0.14, direction: "pushed away from INSUFFICIENT_FUNDS" },
    { feature: "merchant_success_rate", value: "99.1%", shap_value: 0.09, direction: "pushed away from MERCHANT_ISSUE" },
  ],
  recoveryAction: "SMART_RETRY",
  recoveryStatus: "RECOVERED",
  recoveryAttemptedAt: "2026-08-23T10:22:00+05:30",
  actionScores: [
    { action: "SMART_RETRY", score: 0.68, expectedRecovery: 2480 },
    { action: "WHATSAPP_LINK", score: 0.31, expectedRecovery: 1540 },
    { action: "EMAIL_REMINDER", score: 0.16, expectedRecovery: 790 },
  ],
  guardrails: ["Retry only after issuer recovery signal", "One action per 24-hour customer window", "Stop immediately after success"],
};
