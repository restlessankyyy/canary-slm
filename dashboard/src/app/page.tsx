"use client";

import { useState, useEffect } from "react";
import { Activity, ShieldAlert, Server, Radar, ShieldCheck, AlertTriangle } from "lucide-react";

type TransactionResult = {
  transaction_id: string;
  amount: number;
  country: string;
  risk_label: string;
  fraud_probability: number;
  decision_source: string;
  risk_factors: string[];
  action: string;
  processing_time_ms: number;
  timestamp: Date;
};

// Fixed to match FastAPI backend schema exactly
const generateRandomTxn = () => {
  const isHighRisk = Math.random() > 0.85;
  return {
    transaction_id: `TXN-${Math.random().toString(36).substring(2, 8).toUpperCase()}`,
    customer_id: isHighRisk ? (Math.random() > 0.5 ? "CUST_VIP_001" : `CUS_${Math.floor(Math.random() * 90000)}`) : `CUS_${Math.floor(Math.random() * 90000)}`,
    amount: isHighRisk ? Math.random() * 50000 + 1000 : Math.random() * 200,
    merchant_cat: isHighRisk ? "CRYPTO" : "RETAIL",
    country: isHighRisk ? "NG" : "US",
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
          setTransactions((prev) => [newTxn, ...prev].slice(0, 100)); // Keep up to 100 for telemetry
          setStats((s) => ({
            totalScored: s.totalScored + 1,
            blocked: s.blocked + (data.action.includes("Block") || data.action.includes("Decline") ? 1 : 0),
            allowed: s.allowed + (data.action.includes("Approve") || data.action.includes("Allow") ? 1 : 0),
            avgLatency: s.totalScored === 0 ? data.processing_time_ms : (s.avgLatency * 0.9 + data.processing_time_ms * 0.1),
          }));
        })
        .catch(console.error);
    }, 1500); // slightly faster for dense data feeling
    return () => clearInterval(interval);
  }, [isLive]);

  const getRiskColor = (label: string) => {
    if (label.includes("CRITICAL") || label.includes("HIGH RISK")) return "text-[#ff3366] bg-[#ff3366]/10 border-[#ff3366]/30";
    if (label.includes("MEDIUM") || label.includes("REVIEW")) return "text-[#ffcc00] bg-[#ffcc00]/10 border-[#ffcc00]/30";
    return "text-[#00e6b8] bg-[#00e6b8]/10 border-[#00e6b8]/30";
  };

  const highRiskQueue = transactions.filter(t => t.risk_label.includes("CRITICAL") || t.risk_label.includes("MEDIUM") || t.risk_label.includes("REVIEW"));

  return (
    <div className="min-h-screen bg-black text-white font-sans overflow-hidden flex flex-col bg-stars relative selection:bg-[#00e6b8]/30">

      {/* Top Header - Ultra Clean */}
      <header className="h-16 px-6 border-b border-white/10 flex items-center justify-between z-10 bg-black/50 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Radar className="text-white w-5 h-5 animate-[spin_4s_linear_infinite]" />
          <h1 className="text-sm font-bold tracking-widest uppercase">Canary Terminal</h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-zinc-400">
            <span>Server: local-node-01</span>
            <span>|</span>
            <span className={backendStatus === "online" ? "text-[#00e6b8]" : "text-[#ff3366]"}>
              {backendStatus === "online" ? "UPLINK SECURE" : "UPLINK LOST"}
            </span>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            className="text-[10px] font-bold uppercase tracking-widest px-4 py-1.5 border border-white/20 rounded-full hover:bg-white hover:text-black transition-colors"
          >
            {isLive ? "PAUSE FEED" : "RESUME FEED"}
          </button>
        </div>
      </header>

      {/* Metrics Strip */}
      <div className="grid grid-cols-4 border-b border-white/10 z-10 bg-black/50 backdrop-blur-md">
        {[
          { label: "Global Processed", value: stats.totalScored.toLocaleString(), icon: Activity },
          { label: "Threats Blocked", value: stats.blocked.toLocaleString(), icon: ShieldAlert, color: "text-[#ff3366]" },
          { label: "Legitimate Traffic", value: stats.allowed.toLocaleString(), icon: ShieldCheck, color: "text-[#00e6b8]" },
          { label: "Inference Latency", value: `${stats.avgLatency.toFixed(1)}ms`, icon: Server },
        ].map((stat, i) => (
          <div key={i} className="px-6 py-4 flex flex-col border-r border-white/10 last:border-0 hover:bg-white/[0.02] transition-colors">
            <div className="flex justify-between items-center mb-1">
              <span className="text-[10px] uppercase tracking-widest text-zinc-500">{stat.label}</span>
              <stat.icon className={`w-3 h-3 ${stat.color || "text-zinc-500"}`} />
            </div>
            <div className="text-2xl font-light font-mono tracking-tight">{stat.value}</div>
          </div>
        ))}
      </div>

      {/* Main Split Layout */}
      <main className="flex-1 overflow-hidden grid grid-cols-12 z-10">

        {/* LEFT COLUMN: URGENT QUEUE (Dense blocks) - spans 3 columns */}
        <div className="col-span-3 border-r border-white/10 flex flex-col bg-black/40 backdrop-blur-md">
          <div className="px-4 py-3 border-b border-white/10 flex justify-between items-center sticky top-0 bg-black z-20">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-white flex items-center gap-2">
              <AlertTriangle className="w-3 h-3 text-[#ffcc00]" /> Priority Intercept Queue
            </h2>
            <span className="text-[10px] font-mono text-zinc-500">{highRiskQueue.length} Active</span>
          </div>
          <div className="flex-1 overflow-auto custom-scrollbar p-3 space-y-3">
            {highRiskQueue.length === 0 ? (
              <div className="text-[10px] uppercase tracking-widest text-zinc-600 text-center mt-10">
                Queue Clear
              </div>
            ) : (
              highRiskQueue.map((txn, i) => (
                <div key={i} className="border border-white/10 bg-white/[0.02] p-3 rounded-sm hover:bg-white/[0.05] transition-colors cursor-pointer group">
                  <div className="flex justify-between items-start mb-2">
                    <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 border ${getRiskColor(txn.risk_label)}`}>
                      {txn.risk_label.replace(/🟢 |🟡 |🟠 |🔴 |🚨 /g, '')}
                    </span>
                    <span className="text-[10px] font-mono text-zinc-500">{txn.timestamp.toISOString().substring(11, 19)}</span>
                  </div>
                  <div className="flex justify-between items-baseline mb-2">
                    <span className="font-mono text-sm group-hover:text-white transition-colors">{txn.transaction_id}</span>
                    <span className="font-mono text-xs text-white">${txn.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                  </div>
                  <div className="flex justify-between items-center text-[10px]">
                    <span className="uppercase text-zinc-500 tracking-widest">{txn.action}</span>
                    <span className="font-mono text-[#ffcc00] border-b border-[#ffcc00]/30 inline-block">{(txn.fraud_probability * 100).toFixed(1)}% CONF</span>
                  </div>
                  {txn.risk_factors.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-1">
                      {txn.risk_factors.map((rf, idx) => (
                        <span key={idx} className="text-[9px] font-mono uppercase bg-white/5 px-1 py-0.5 text-zinc-400">
                          {rf}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: FULL TELEMETRY (Data Table) - spans 9 columns */}
        <div className="col-span-9 flex flex-col bg-black/20 backdrop-blur-md">
          <div className="px-6 py-3 border-b border-white/10 flex justify-between items-center sticky top-0 bg-black z-20">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-white">Full Network Telemetry</h2>
            <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
              Live Feed
            </span>
          </div>

          <div className="flex-1 overflow-auto custom-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 bg-black z-10 text-[9px] font-bold uppercase tracking-widest text-zinc-500 border-b border-white/10">
                <tr>
                  <th className="px-6 py-3 font-normal">Timestamp</th>
                  <th className="px-6 py-3 font-normal">Identifier</th>
                  <th className="px-6 py-3 font-normal text-right">Value (USD)</th>
                  <th className="px-6 py-3 font-normal text-center">Node</th>
                  <th className="px-6 py-3 font-normal">Engine Path</th>
                  <th className="px-6 py-3 font-normal">Confidence</th>
                  <th className="px-6 py-3 font-normal">Execution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.03]">
                {transactions.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-[10px] uppercase tracking-widest text-zinc-600">
                      Waiting for stream...
                    </td>
                  </tr>
                ) : (
                  transactions.map((txn, i) => (
                    <tr key={i} className={`hover:bg-white/[0.03] transition-colors ${i === 0 ? 'bg-white/[0.02]' : ''}`}>
                      <td className="px-6 py-3 text-xs text-zinc-500 font-mono whitespace-nowrap">
                        {txn.timestamp.toISOString().substring(11, 23)}
                      </td>
                      <td className="px-6 py-3 font-mono text-xs text-zinc-300">
                        {txn.transaction_id}
                      </td>
                      <td className="px-6 py-3 font-mono text-xs text-white text-right">
                        ${txn.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td className="px-6 py-3 text-center">
                        <span className="text-[10px] font-mono text-zinc-400">
                          {txn.country}
                        </span>
                      </td>
                      <td className="px-6 py-3">
                        <span className="text-[9px] uppercase tracking-widest text-zinc-500">
                          {txn.decision_source}
                        </span>
                      </td>
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] font-bold uppercase tracking-widest w-16 ${getRiskColor(txn.risk_label).split(' ')[0]}`}>
                            {txn.risk_label.replace(/🟢 |🟡 |🟠 |🔴 |🚨 /g, '')}
                          </span>
                          <span className="text-[10px] font-mono text-zinc-500 w-10 text-right">
                            {(txn.fraud_probability * 100).toFixed(1)}%
                          </span>
                          <div className="w-16 h-1 bg-white/10 relative rounded-none overflow-hidden hidden xl:block">
                            <div
                              className={`absolute top-0 left-0 h-full ${txn.fraud_probability > 0.7 ? 'bg-[#ff3366]' : txn.fraud_probability > 0.3 ? 'bg-[#ffcc00]' : 'bg-[#00e6b8]'}`}
                              style={{ width: `${Math.max(2, txn.fraud_probability * 100)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-3">
                        <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-300">
                          {txn.action}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

      </main>
    </div>
  );
}
