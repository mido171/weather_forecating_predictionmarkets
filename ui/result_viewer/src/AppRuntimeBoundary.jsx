import React from "react";

const AUTO_RETRY_DELAY_MS = 300;

export default class AppRuntimeBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      error: null,
      resetKey: 0,
      autoRetried: false,
    };
    this.retryTimer = null;
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("UI runtime error", error, errorInfo);
    if (!this.state.autoRetried) {
      this.retryTimer = window.setTimeout(() => {
        this.setState((current) => ({
          error: null,
          resetKey: current.resetKey + 1,
          autoRetried: true,
        }));
      }, AUTO_RETRY_DELAY_MS);
    }
  }

  componentWillUnmount() {
    if (this.retryTimer) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  handleRetry = () => {
    if (this.retryTimer) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.setState((current) => ({
      error: null,
      resetKey: current.resetKey + 1,
      autoRetried: false,
    }));
  };

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: "100vh",
            display: "grid",
            placeItems: "center",
            padding: "24px",
            background: "#f4efe4",
            color: "#24180f",
          }}
        >
          <div
            style={{
              width: "min(560px, 100%)",
              border: "1px solid #d3c2ab",
              background: "#fffaf2",
              borderRadius: "18px",
              padding: "24px",
              boxShadow: "0 20px 60px rgba(36, 24, 15, 0.12)",
            }}
          >
            <h1 style={{ margin: "0 0 10px", fontSize: "1.25rem" }}>UI runtime failure</h1>
            <p style={{ margin: "0 0 16px", lineHeight: 1.5 }}>
              A transient client-side error interrupted rendering. The app already attempted one automatic recovery.
            </p>
            <pre
              style={{
                margin: "0 0 16px",
                padding: "12px",
                borderRadius: "12px",
                background: "#f6ede0",
                overflowX: "auto",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                fontSize: "0.9rem",
              }}
            >
              {String(this.state.error?.message ?? this.state.error ?? "Unknown error")}
            </pre>
            <button
              type="button"
              onClick={this.handleRetry}
              style={{
                border: "none",
                borderRadius: "999px",
                padding: "10px 16px",
                background: "#24180f",
                color: "#fffaf2",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Retry UI
            </button>
          </div>
        </div>
      );
    }

    return <React.Fragment key={this.state.resetKey}>{this.props.children}</React.Fragment>;
  }
}
