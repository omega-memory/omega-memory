/* Google Identity Services (GIS) types */

interface TokenResponse {
  access_token: string;
  expires_in: number;
  scope: string;
  token_type: string;
  error?: string;
  error_description?: string;
}

interface TokenClient {
  requestAccessToken(config?: { prompt?: string }): void;
}

interface TokenClientConfig {
  client_id: string;
  scope: string;
  callback: (response: TokenResponse) => void;
  error_callback?: (error: { type: string; message: string }) => void;
}

declare namespace google {
  namespace accounts {
    namespace oauth2 {
      function initTokenClient(config: TokenClientConfig): TokenClient;
    }
  }

  namespace picker {
    class DocsView {
      constructor();
      setMimeTypes(mimeTypes: string): DocsView;
      setIncludeFolders(include: boolean): DocsView;
    }

    class PickerBuilder {
      addView(view: DocsView): PickerBuilder;
      setOAuthToken(token: string): PickerBuilder;
      setDeveloperKey(key: string): PickerBuilder;
      setAppId(appId: string): PickerBuilder;
      enableFeature(feature: Feature): PickerBuilder;
      setCallback(callback: (data: PickerCallbackData) => void): PickerBuilder;
      build(): Picker;
    }

    interface Picker {
      setVisible(visible: boolean): void;
    }

    interface PickerCallbackData {
      action: string;
      docs?: PickerDocument[];
    }

    interface PickerDocument {
      id: string;
      name: string;
      mimeType: string;
      sizeBytes?: number;
    }

    enum Feature {
      MULTISELECT_ENABLED = "multiselect",
    }
  }
}

declare namespace gapi {
  function load(api: string, callback: () => void): void;
}
