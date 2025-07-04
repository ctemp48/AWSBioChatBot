export function getSessionId(): string {
    if (typeof window === 'undefined') return ''; // Server guard
  
    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = crypto.randomUUID(); // Safe in browser
      localStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }

export function clearSessionId(): void {
  if (typeof window === 'undefined') return; // Server-side guard

  localStorage.removeItem('session_id');
}
  