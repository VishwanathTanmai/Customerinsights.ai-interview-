"use client";
import { useEffect, useState } from "react";

const KEY = "uplift.rep_id";
const EVT = "uplift-rep-change";

export function useRep(): [string, (v: string) => void] {
  const [rep, setRepState] = useState("");
  useEffect(() => {
    setRepState(localStorage.getItem(KEY) || "");
    const h = () => setRepState(localStorage.getItem(KEY) || "");
    window.addEventListener(EVT, h);
    return () => window.removeEventListener(EVT, h);
  }, []);
  const setRep = (v: string) => {
    localStorage.setItem(KEY, v);
    window.dispatchEvent(new Event(EVT));
  };
  return [rep, setRep];
}
