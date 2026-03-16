import { useState, useEffect, useCallback } from "react";

interface UserInfo {
  user: { id: string; email: string } | null;
  role: "owner" | "contributor" | null;
  loading: boolean;
  error: string | null;
}

export function useCurrentUser(): UserInfo {
  const [info, setInfo] = useState<UserInfo>({ user: null, role: null, loading: true, error: null });

  const fetchUser = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/me");
      if (!res.ok) {
        if (res.status === 401) {
          setInfo({ user: null, role: null, loading: false, error: null });
          return;
        }
        if (res.status === 403) {
          const data = await res.json().catch(() => ({}));
          if (data.code === "pro_required") {
            window.location.href = "/admin/login?error=pro_required";
            return;
          }
        }
        setInfo({ user: null, role: null, loading: false, error: `HTTP ${res.status}` });
        return;
      }
      const data = await res.json();
      setInfo({
        user: data.user ?? null,
        role: data.role ?? null,
        loading: false,
        error: null,
      });
    } catch (err) {
      setInfo({
        user: null,
        role: null,
        loading: false,
        error: err instanceof Error ? err.message : "Failed to load user",
      });
    }
  }, []);

  useEffect(() => {
    fetchUser();

    // Refetch when tab becomes visible
    function handleVisibility() {
      if (!document.hidden) {
        fetchUser();
      }
    }
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [fetchUser]);

  return info;
}
