"use client";

import { useState, useEffect } from "react";
import { Activity, AlertTriangle, CheckCircle, ShieldAlert, ShieldCheck, Clock, RefreshCw, XCircle } from "lucide-react";

type TransactionResult = {
  transaction_id: string;
  amount: number;
  country: string;
  risk_label: string;
  fraud_probability: number;
  decision_source: string;
  risk_factors: string[];
  processing_time_ms: number;
  timestamp: Date;
};

// Helper to generate a random synthetic transaction payload
const generateRandomTxn = () => {
  const isHighRisk = Math.random() > 0.8;
  return {
    transaction_id: `txn_${Math.random().toString(36).substring(2, 9)}`,
    customer_id: isHighRisk ? (Math.random() > 0.5 ? "CUST_VIP_001" : `CUST_${Math.floor(Math.random() * 90000)}`) : `CUST_${Math.floor(Math.random() * 90000)}`,
    amount: isHighRisk ? Math.random() * 100000 : Math.random() * 500,
    merchant_cat: isHighRisk ? "CRYPTO" : "RETAIL",
    country: isHighRisk ? (Math.random() > 0.5 ? "NG" : "KP") : "US",
    is_domestic: !isHighRisk,
    hour: isHighRisk ? 3 : 12,
    day_of_week: 3,
    channel: "ONLINE",
    currency: "USD",
    velocity: isHighRisk ? "EXTREME" : "NORMAL",
    flags: isHighRisk ? ["FOREIGN_IP", "NEW_DEVICE"] : [],
  };
};

export default function Dashboard() {
  const [transactions, setTransactions] = useState<TransactionResult[]>([]);
  const [isLive, setIsLive] = useState(true);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "online" | "offline">("connecting");

  const [stats, setStats] = useState({
    totalScored: 0,
    blocked: 0,
    allowed: 0,
    avgLatency: 0,
  });

  useEffect(() => {
    // Check backend health
    fetch("http://localhost:8000/health")
      .then((res) => (res.ok ? setBackendStatus("online") : setBackendStatus("offline")))
      .catch(() => setBackendStatus("offline"));

    if (!isLive) return;

    const interval = setInterval(() => {
      const payload = generateRandomTxn();

      fetch("http://localhost:8000/v1/score/fraud", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
        .then((res) => res.json())
        .then((data) => {
          if (!data.risk_label) return; // API offline or error

          const newTxn: TransactionResult = {
            ...data,
            amount: payload.amount,
            country: payload.country,
            timestamp: new Date(),
          };

          setTransactions((prev) => [newTxn, ...prev].slice(0, 50));

          setStats((s) => ({
            totalScored: s.totalScored + 1,
            blocked: s.blocked + (data.action.includes("Block") || data.action.includes("Decline") ? 1 : 0),
            allowed: s.allowed + (data.action.includes("Approve") || data.action.includes("Allow") ? 1 : 0),
            avgLatency: s.totalScored === 0 ? data.processing_time_ms : (s.avgLatency * 0.9 + data.processing_time_ms * 0.1),
          }));
        })
        .catch(console.error);
    }, 2500);

    return () => clearInterval(interval);
  }, [isLive]);

  const getRiskIcon = (label: string) => {
    if (label.includes("CRITICAL") || label.includes("HIGH RISK")) return <XCircle className="w-5 h-5 text-red-500" />;
    if (label.includes("MEDIUM") || label.includes("REVIEW")) return <AlertTriangle className="w-5 h-5 text-amber-500" />;
    return <CheckCircle className="w-5 h-5 text-emerald-500" />;
  };

  const getRiskBadge = (label: string) => {
    if (label.includes("CRITICAL") || label.includes("HIGH RISK")) return "bg-red-500/10 text-red-400 border border-red-500/20";
    if (label.includes("MEDIUM") || label.includes("REVIEW")) return "bg-amber-500/10 text-amber-400 border border-amber-500/20";
    return "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
  };

  return (
    <div className="min-h-screen p-8 selection:bg-indigo-500/30">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 animate-slide-in">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <ShieldAlert className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              CANARY Enterprise
            </h1>
            <p className="text-sm text-slate-400">Fraud Operations Center</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 glass-panel px-4 py-2 rounded-full text-sm">
            <div className={`w-2 h-2 rounded-full ${backendStatus === "online" ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
            <span className="text-slate-300">FastAPI Gateway: {backendStatus.toUpperCase()}</span>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${isLive ? "bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/20" : "glass-panel text-slate-300 hover:text-white"
              }`}
          >
            {isLive ? <Activity className="w-4 h-4 animate-pulse" /> : <RefreshCw className="w-4 h-4" />}
            {isLive ? "Live Stream Active" : "Stream Paused"}
          </button>
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[
          { label: "Transactions Scored", value: stats.totalScored.toLocaleString(), icon: Activity, color: "text-blue-400" },
          { label: "Threats Blocked", value: stats.blocked.toLocaleString(), icon: ShieldCheck, color: "text-red-400" },
          { label: "Allowed Traffic", value: stats.allowed.toLocaleString(), icon: CheckCircle, color: "text-emerald-400" },
          { label: "Avg Inference Latency", value: `${stats.avgLatency.toFixed(1)} ms`, icon: Clock, color: "text-purple-400" },
        ].map((stat, i) => (
          <div key={i} className="glass-panel p-6 rounded-2xl animate-slide-in" style={{ animationDelay: `${i * 100}ms` }}>
            <div className="flex justify-between items-start mb-4">
              <stat.icon className={`w-6 h-6 ${stat.color}`} />
            </div>
            <h3 className="text-3xl font-bold text-white mb-1">{stat.value}</h3>
            <p className="text-sm text-slate-400 font-medium">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Live Feed */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-slate-700/50 flex flex-col h-[600px] animate-slide-in" style={{ animationDelay: "400ms" }}>
        <div className="glass-header p-5 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" /> Live Transaction Stream
          </h2>
        </div>

        <div className="overflow-auto flex-1 p-0 custom-scrollbar">
          <table className="w-full text-left border-collapse">
            <thead className="sticky top-0 glass-header z-10 text-xs uppercase font-semibold text-slate-400">
              <tr>
                <th className="px-6 py-4">Time</th>
                <th className="px-6 py-4">Transaction</th>
                <th className="px-6 py-4">Amount</th>
                <th className="px-6 py-4">Geo</th>
                <th className="px-6 py-4">Risk Decision</th>
                <th className="px-6 py-4">AI Confidence</th>
                <th className="px-6 py-4">Action Taken</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {transactions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                    {backendStatus === "online" ? "Waiting for transactions..." : "FastAPI Backend is offline. Run 'docker-compose up'."}
                  </td>
                </tr>
              ) : (
                transactions.map((txn, i) => (
                  <tr key={txn.transaction_id} className={`hover:bg-slate-800/30 transition-colors ${i === 0 ? 'bg-indigo-500/5' : ''}`}>
                    <td className="px-6 py-4 text-sm text-slate-400 whitespace-nowrap">
                      {txn.timestamp.toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4">
                      <div className="font-mono text-xs text-slate-300">{txn.transaction_id}</div>
                      <div className="text-xs text-slate-500 mt-1">{txn.decision_source}</div>
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-slate-200">
                      ${txn.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-6 py-4">
                      <span className="px-2.5 py-1 rounded bg-slate-800 text-xs font-semibold text-slate-300 border border-slate-700">
                        {txn.country}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {getRiskIcon(txn.risk_label)}
                        <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${getRiskBadge(txn.risk_label)}`}>
                          {txn.risk_label.replace(/🟢 |🟡 |🟠 |🔴 |🚨 /g, '')}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${txn.fraud_probability > 0.7 ? 'bg-red-500' : txn.fraud_probability > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                            style={{ width: `${Math.max(5, txn.fraud_probability * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-400 font-mono">
                          {(txn.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-slate-300">
                        {txn.action}
                      </div>
                      {txn.risk_factors.length > 0 && (
                        <div className="text-xs text-slate-500 mt-1 truncate max-w-[200px]" title={txn.risk_factors.join(", ")}>
                          {txn.risk_factors[0]}
                          {txn.risk_factors.length > 1 && ` +${txn.risk_factors.length - 1} more`}
                        </div>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
