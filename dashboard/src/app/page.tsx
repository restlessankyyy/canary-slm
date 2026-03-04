"use client";

import { useState, useEffect } from "react";
import { Activity, ShieldAlert, Zap, Server, ActivitySquare } from "lucide-react";

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

const generateRandomTxn = () => {
  const isHighRisk = Math.random() > 0.8;
  return {
    transaction_id: `txn_${Math.random().toString(36).substring(2, 9).toUpperCase()}`,
    customer_id: isHighRisk ? (Math.random() > 0.5 ? "CUST_VIP_001" : `CUS_${Math.floor(Math.random() * 90000)}`) : `CUS_${Math.floor(Math.random() * 90000)}`,
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
          if (!data.risk_label) return;
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
    }, 2000);
    return () => clearInterval(interval);
  }, [isLive]);

  const getRiskColor = (label: string) => {
    if (label.includes("CRITICAL") || label.includes("HIGH RISK")) return "text-[#ff3366] border-[#ff3366]";
    if (label.includes("MEDIUM") || label.includes("REVIEW")) return "text-[#ffcc00] border-[#ffcc00]";
    return "text-[#00e6b8] border-[#00e6b8]";
  };

  return (
    <div className="min-h-screen p-6 md:p-12 selection:bg-[#00e6b8]/20 relative overflow-hidden">
      {/* Background ambient glow */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-[#00e6b8] opacity-[0.03] blur-[150px] rounded-full pointer-events-none" />

      {/* Header */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between mb-16 animate-slide-up opacity-0" style={{ animationDelay: "0ms" }}>
        <div className="flex items-center gap-4 mb-6 md:mb-0">
          <div className="w-12 h-12 bg-black border border-white/10 flex items-center justify-center">
            <ShieldAlert className="text-[#00e6b8] w-6 h-6 glow-text-cyan" />
          </div>
          <div>
            <h1 className="text-3xl font-light tracking-tight text-white m-0 leading-none">
              CANARY <span className="font-semibold">CORE</span>
            </h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-500 mt-2">Autonomous Risk Operations</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <span className="text-[10px] uppercase tracking-widest-xs text-zinc-500">System Link</span>
            <div className={`flex items-center gap-2 px-3 py-1.5 border ${backendStatus === "online" ? "border-glow-cyan text-[#00e6b8]" : "border-red-500/30 text-red-500"} text-xs font-mono-data`}>
              <div className={`w-1.5 h-1.5 rounded-full ${backendStatus === "online" ? "bg-[#00e6b8] animate-pulse-slow" : "bg-red-500"}`} />
              {backendStatus === "online" ? "ACTIVE" : "DISCONNECTED"}
            </div>
          </div>

          <button
            onClick={() => setIsLive(!isLive)}
            className={`flex items-center gap-2 px-5 py-2 text-[10px] uppercase tracking-widest-xs transition-all border ${isLive
                ? "bg-white text-black border-white hover:bg-zinc-200"
                : "bg-transparent text-white border-white/20 hover:border-white/50"
              }`}
          >
            {isLive ? <ActivitySquare className="w-3 h-3 animate-pulse" /> : <Server className="w-3 h-3" />}
            {isLive ? "Stream Live" : "Stream Suspended"}
          </button>
        </div>
      </header>

      {/* Telemetry Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-16">
        {[
          { label: "Total Inferences", value: stats.totalScored.toLocaleString(), icon: Zap },
          { label: "Threats Intercepted", value: stats.blocked.toLocaleString(), icon: ShieldAlert },
          { label: "Cleared Traffic", value: stats.allowed.toLocaleString(), icon: Activity },
          { label: "Compute Latency", value: `${stats.avgLatency.toFixed(1)}ms`, icon: Server },
        ].map((stat, i) => (
          <div key={i} className="glass-panel p-6 animate-slide-up opacity-0" style={{ animationDelay: `${(i + 1) * 100}ms` }}>
            <div className="flex justify-between items-start mb-6">
              <span className="text-[10px] uppercase tracking-widest-xs text-zinc-500">{stat.label}</span>
              <stat.icon className="w-4 h-4 text-zinc-700" />
            </div>
            <div className="text-4xl font-light text-white font-mono-data tracking-tight">
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Data Grid */}
      <div className="animate-slide-up opacity-0 relative" style={{ animationDelay: "500ms" }}>
        <div className="flex justify-between items-end mb-4">
          <h2 className="text-[10px] uppercase tracking-[0.2em] text-zinc-400">Live Telemetry Feed</h2>
          <div className="text-[10px] text-zinc-600 font-mono-data">UTC {new Date().toISOString().substring(11, 19)}</div>
        </div>

        <div className="glass-panel border-t border-b border-l-0 border-r-0 border-white/10 flex flex-col h-[500px]">
          <div className="overflow-auto flex-1 custom-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 bg-[#000000] z-10 text-[10px] uppercase tracking-widest-xs text-zinc-600 border-b border-white/5">
                <tr>
                  <th className="px-6 py-4 font-normal">Time</th>
                  <th className="px-6 py-4 font-normal">TXN Hash</th>
                  <th className="px-6 py-4 font-normal text-right">Value (USD)</th>
                  <th className="px-6 py-4 font-normal text-center">Node</th>
                  <th className="px-6 py-4 font-normal">Classification</th>
                  <th className="px-6 py-4 font-normal">AI Match</th>
                  <th className="px-6 py-4 font-normal">Execution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {transactions.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-20 text-center text-zinc-600 text-sm font-light">
                      {backendStatus === "online" ? "Awaiting telemetry data..." : "SYSTEM OFFLINE."}
                    </td>
                  </tr>
                ) : (
                  transactions.map((txn, i) => (
                    <tr key={txn.transaction_id} className={`hover:bg-white/[0.02] transition-colors group ${i === 0 ? 'bg-[#00e6b8]/[0.02]' : ''}`}>
                      <td className="px-6 py-4 text-xs text-zinc-500 font-mono-data whitespace-nowrap">
                        {txn.timestamp.toISOString().substring(11, 23)}
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-mono-data text-sm text-zinc-300">{txn.transaction_id}</div>
                        <div className="text-[10px] text-zinc-600 mt-1 uppercase tracking-wider">{txn.decision_source.replace(/_/g, ' ')}</div>
                      </td>
                      <td className="px-6 py-4 font-mono-data text-sm text-white text-right">
                        ${txn.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span className="text-xs font-mono-data text-zinc-400">
                          {txn.country}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-[10px] uppercase tracking-widest-xs font-semibold ${getRiskColor(txn.risk_label).split(' ')[0]}`}>
                          {txn.risk_label.replace(/🟢 |🟡 |🟠 |🔴 |🚨 /g, '')}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-zinc-400 font-mono-data w-12 text-right">
                            {(txn.fraud_probability * 100).toFixed(1)}%
                          </span>
                          <div className="w-24 h-[1px] bg-white/10 relative">
                            <div
                              className={`absolute top-0 left-0 h-full ${txn.fraud_probability > 0.7 ? 'bg-[#ff3366] shadow-[0_0_8px_#ff3366]' : txn.fraud_probability > 0.3 ? 'bg-[#ffcc00]' : 'bg-[#00e6b8] shadow-[0_0_8px_rgba(0,230,184,0.5)]'}`}
                              style={{ width: `${Math.max(2, txn.fraud_probability * 100)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-xs text-zinc-300">
                          {txn.action}
                        </div>
                        {txn.risk_factors.length > 0 && (
                          <div className="text-[10px] tracking-wider uppercase text-zinc-600 mt-1 truncate max-w-[200px]" title={txn.risk_factors.join(", ")}>
                            {txn.risk_factors.join(" · ")}
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
    </div>
  );
}
