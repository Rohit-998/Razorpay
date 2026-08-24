import { PaymentInsights } from "../../../components/payment-insights";

export default async function PaymentInsightsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <PaymentInsights paymentId={id} />;
}
