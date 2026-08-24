"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Brain,
  Target,
  Shield,
  Zap,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  CreditCard,
  Info,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PaymentData {
  payment_id: string;
  amount: number;
  method: string;
  bank: string | null;
  error_code: string;
  error_source: string;
  error_reason: string;
  error_description: string;
  customer_contact: string | null;
  customer_email: string | null;
  created_at: string;
}

interface ShapExplanation {
  feature: string;
  value: string | number;
  shap_value: number;
  direction: string;
}

interface SessionData {
  id: string;
  status: string;
  root_cause: string;
  root_cause_confidence: number | null;
  strategy: string;
  decided_by: string;
  amount_recovered: number;
  shap_explanation: ShapExplanation[] | null;
  attribution: string | null;
  closed_at: string | null;
}

interface AuditEvent {
  event_type: string;
  event_data: Record<string, unknown>;
  created_at: string;
}

export default function PaymentDetailPage() {
  const params = useParams();
  const paymentId = params.id as string;

  const [payment, setPayment] = useState<PaymentData | null>(null);
  const [session, setSession] = useState<SessionData | null>(null);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/payments/${paymentId}`)
      .then((r) => r.json())
      .then((data) => {
        setPayment(data.payment);
        setSession(data.session);
        setAudit(data.audit_trail || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [paymentId]);

  const formatAmount = (paise: number) =>
    `₹${(paise / 100).toLocaleString("en-IN")}`;

  const getStatusBadge = (status: string) => {
    const map: Record<string, { icon: React.ReactNode; cls: string }> = {
      RECOVERED: {
        icon: <CheckCircle2 className="w-4 h-4" />,
        cls: "text-green-400 bg-green-500/15 border-green-500/30",
      },
      FAILED: {
        icon: <XCircle className="w-4 h-4" />,
        cls: "text-red-400 bg-red-500/15 border-red-500/30",
      },
      ESCALATED: {
        icon: <AlertTriangle className="w-4 h-4" />,
        cls: "text-yellow-400 bg-yellow-500/15 border-yellow-500/30",
      },
      OPEN: {
        icon: <Clock className="w-4 h-4" />,
        cls: "text-gray-400 bg-gray-500/15 border-gray-500/30",
      },
    };
    const style = map[status] || map.OPEN;
    return (
      <span
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium border ${style.cls}`}
      >
        {style.icon}
        {status}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="p-8">
        <div className="h-8 w-48 shimmer rounded mb-4" />
        <div className="grid grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="h-64 shimmer rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  if (!payment) {
    return (
      <div className="p-8">
        <p className="text-gray-500">Payment not found.</p>
      </div>
    );
  }

  // Prepare SHAP data for chart
  const shapData = (session?.shap_explanation || [])
    .map((s) => ({
      feature: s.feature.replace(/_/g, " "),
      value: Math.abs(s.shap_value),
      original: s.shap_value,
      rawValue: s.value,
      fill: s.shap_value > 0 ? "#ef4444" : "#3b82f6",
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const confidence = session?.root_cause_confidence
    ? Math.round(session.root_cause_confidence * 100)
    : 0;

  return (
    <div className="p-8">
      {/* Breadcrumb */}
      <div className="mb-6">
        <Link
          href="/payments"
          className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Payments
        </Link>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold font-mono mb-1">
            {payment.payment_id}
          </h1>
          <p className="text-gray-500 text-sm">
            {new Date(payment.created_at).toLocaleString()} ·{" "}
            {formatAmount(payment.amount)}
          </p>
        </div>
        {session && getStatusBadge(session.status)}
      </div>

      {/* AI Pipeline Steps */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Step 1: Transaction Context */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center">
              <CreditCard className="w-4 h-4 text-gray-400" />
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">
                Step 1
              </p>
              <h3 className="text-sm font-semibold">Transaction Context</h3>
            </div>
          </div>
          <div className="space-y-3">
            <DetailRow label="Amount" value={formatAmount(payment.amount)} />
            <DetailRow
              label="Method"
              value={payment.method.toUpperCase()}
            />
            <DetailRow label="Bank" value={payment.bank || "N/A"} />
            <DetailRow label="Error Code" value={payment.error_code} />
            <DetailRow label="Error Source" value={payment.error_source} />
            <DetailRow
              label="Error Reason"
              value={payment.error_reason.replace(/_/g, " ")}
              highlight
            />
          </div>
        </div>

        {/* Step 2: XGBoost Prediction */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-blue-500/15 flex items-center justify-center">
              <Brain className="w-4 h-4 text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-blue-400 uppercase tracking-wide">
                Step 2
              </p>
              <h3 className="text-sm font-semibold">
                XGBoost Root Cause Prediction
              </h3>
            </div>
          </div>
          {session?.root_cause ? (
            <>
              <div className="bg-gray-950 border border-gray-800 rounded-lg p-4 mb-4">
                <p className="text-xs text-gray-500 mb-1">
                  Predicted Root Cause
                </p>
                <p className="text-lg font-bold text-orange-400">
                  {session.root_cause.replace(/_/g, " ")}
                </p>
              </div>
              <div className="bg-gray-950 border border-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-500 mb-2">
                  Confidence Score
                </p>
                <div className="flex items-end gap-2">
                  <p className="text-3xl font-bold text-blue-400">
                    {confidence}%
                  </p>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2 mt-3">
                  <div
                    className="h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${confidence}%`,
                      backgroundColor:
                        confidence > 80
                          ? "#22c55e"
                          : confidence > 50
                          ? "#eab308"
                          : "#ef4444",
                    }}
                  />
                </div>
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-sm">Not yet classified.</p>
          )}
        </div>

        {/* Step 3: Bandit Action */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 flex items-center justify-center">
              <Target className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <p className="text-xs text-indigo-400 uppercase tracking-wide">
                Step 3
              </p>
              <h3 className="text-sm font-semibold">
                Contextual Bandit Decision
              </h3>
            </div>
          </div>
          {session?.strategy ? (
            <>
              <div className="bg-gray-950 border border-gray-800 rounded-lg p-4 mb-4">
                <p className="text-xs text-gray-500 mb-1">
                  Selected Strategy
                </p>
                <p className="text-lg font-bold text-cyan-400">
                  {session.strategy.replace(/_/g, " ")}
                </p>
              </div>
              <div className="space-y-3">
                <DetailRow
                  label="Decided By"
                  value={session.decided_by || "System"}
                />
                <DetailRow
                  label="Attribution"
                  value={session.attribution?.replace(/_/g, " ") || "—"}
                />
                <DetailRow
                  label="Amount Recovered"
                  value={
                    session.amount_recovered
                      ? formatAmount(session.amount_recovered)
                      : "₹0"
                  }
                  highlight={session.amount_recovered > 0}
                />
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-sm">No strategy selected.</p>
          )}
        </div>
      </div>

      {/* SHAP Explanations */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-8 h-8 rounded-lg bg-purple-500/15 flex items-center justify-center">
            <Shield className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">
              SHAP Explainability — Why did the AI make this decision?
            </h3>
            <p className="text-xs text-gray-500">
              Feature importance values from the XGBoost model
            </p>
          </div>
        </div>

        {shapData.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Chart */}
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shapData}
                  layout="vertical"
                  margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
                >
                  <XAxis
                    type="number"
                    stroke="#6b7280"
                    fontSize={11}
                    tickFormatter={(v) => v.toFixed(2)}
                  />
                  <YAxis
                    dataKey="feature"
                    type="category"
                    stroke="#6b7280"
                    width={140}
                    fontSize={11}
                    tick={{ fill: "#9ca3af" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#111827",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number, name: string, props: any) => [
                      `SHAP: ${props?.payload?.rawValue !== undefined ? Number(props.payload.rawValue).toFixed(4) : value.toFixed(4)}`,
                      "Impact",
                    ]}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {shapData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Feature List */}
            <div className="space-y-2">
              <div className="flex items-center gap-4 mb-3 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-blue-500 rounded-sm inline-block" />{" "}
                  Supports prediction
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-red-500 rounded-sm inline-block" />{" "}
                  Against prediction
                </span>
              </div>
              {(session?.shap_explanation || []).slice(0, 8).map((s, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between px-3 py-2 bg-gray-950 rounded-lg border border-gray-800/50"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-1.5 h-8 rounded-full ${
                        s.shap_value > 0 ? "bg-red-500" : "bg-blue-500"
                      }`}
                    />
                    <div>
                      <p className="text-xs font-medium">
                        {s.feature.replace(/_/g, " ")}
                      </p>
                      <p className="text-[10px] text-gray-500">
                        value: {String(s.value)}
                      </p>
                    </div>
                  </div>
                  <span
                    className={`text-xs font-mono font-medium ${
                      s.shap_value > 0 ? "text-red-400" : "text-blue-400"
                    }`}
                  >
                    {s.shap_value > 0 ? "+" : ""}
                    {s.shap_value.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-gray-500 text-sm py-8 text-center">
            No SHAP explanations available for this payment.
          </p>
        )}
      </div>

      {/* Audit Trail */}
      {audit.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Info className="w-4 h-4 text-gray-400" />
            Audit Trail
          </h3>
          <div className="space-y-3">
            {audit.map((e, i) => (
              <div
                key={i}
                className="flex items-start gap-4 px-4 py-3 bg-gray-950 rounded-lg border border-gray-800/50"
              >
                <div className="w-2 h-2 bg-blue-400 rounded-full mt-1.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-blue-400">
                      {e.event_type.replace(/_/g, " ")}
                    </span>
                    <span className="text-[10px] text-gray-600">
                      {new Date(e.created_at).toLocaleString()}
                    </span>
                  </div>
                  <pre className="text-[10px] text-gray-500 overflow-x-auto whitespace-pre-wrap">
                    {JSON.stringify(e.event_data, null, 2)}
                  </pre>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DetailRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <span
        className={`text-xs font-medium ${
          highlight ? "text-red-400" : "text-gray-300"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
