import { useCallback, useEffect, useState } from "react";

interface State<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

/** Fetch once per key, with a manual reload for after a mutation.
 *
 * Guards against the stale-response race: a reload triggered while an earlier
 * request is still in flight must not have the older response overwrite the
 * newer one. In the review view that would silently resurrect a verdict the
 * operator has already cleared.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): State<T> & {
  reload: () => void;
} {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(fn, deps);

  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true }));
    run()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error: Error) => live && setState({ data: null, error, loading: false }));
    return () => {
      live = false;
    };
  }, [run, nonce]);

  return { ...state, reload: () => setNonce((n) => n + 1) };
}
