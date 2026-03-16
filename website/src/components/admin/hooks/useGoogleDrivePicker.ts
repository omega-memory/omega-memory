// Stub — Google Drive integration not available in self-hosted mode
export interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  webViewLink?: string;
}

export function useGoogleDrivePicker() {
  return {
    openPicker: () => {
      console.warn("Google Drive picker not available in self-hosted mode");
    },
    isLoading: false,
  };
}
