import { PaymentInsights } from "../../../components/payment-insights";

export default function PaymentInsightsPage({ params }: { params: { id: string } }) {
  return <PaymentInsights paymentId={params.id} />;
}
