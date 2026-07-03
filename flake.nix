{
  description = "keybo — data-driven keyboard layout optimizer";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # Native libs that PyPI wheels (numpy, scipy, scikit-learn, xgboost, matplotlib)
      # dlopen at runtime. On NixOS there is no /usr/lib, so these must be on the loader
      # path or the wheels fail to import with "cannot open shared object file".
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib # libstdc++.so.6, libgcc
        gcc.cc.lib       # libgomp.so.1 (OpenMP — xgboost, scikit-learn)
        zlib             # libz.so.1
        glib             # libglib-2.0 (matplotlib backends)
        libGL            # libGL.so.1 (matplotlib)
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv    # fast venv + pip
          just  # task runner
          ruff  # lint + format
          git
        ];

        # Put the native libs above on the loader path so the CPU-only PyPI wheels load.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

        # If this host has the nix-ld NixOS module enabled, these give the same glue to
        # programs run outside this shell. Harmless if nix-ld is not installed.
        NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
        NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

        shellHook = ''
          echo "keybo dev shell · $(python --version)"
          if [ ! -d .venv ]; then
            echo "Creating .venv (uv, python 3.11)…"
            uv venv --python "$(command -v python3.11)" .venv
          fi
          source .venv/bin/activate
          echo "venv active. First time: 'just install'. Then 'just doctor'."
        '';
      };
    };
}
