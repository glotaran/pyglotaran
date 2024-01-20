{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
      python = pkgs.python310;
      lib-path = with pkgs; lib.makeLibraryPath [
        stdenv.cc.cc
        zlib
      ];
      ld-path = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";



    in
    rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python
          pre-commit
          (python3.withPackages (ps: with ps; [
            pip
            pytz
            virtualenvwrapper
          ]))
          zlib
        ];
        shellHook = ''
          # Augment the dynamic linker path
          export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
          export "NIX_LD_LIBRARY_PATH=${lib-path}"
          export "NIX_LD=${ld-path}"

          # Setup the virtual environment if it doesn't already exist.
          export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
          VENV=.venv
          if test ! -d $VENV; then
            virtualenv $VENV
            source ./$VENV/bin/activate
            # Install Python dependencies
            pip install -r requirements_pinned.txt
            pip install -e .[dev]
          fi
          source ./$VENV/bin/activate

          pre-commit install
        '';

      };
    }
  );
}
