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
      python = pkgs.python310;
      lib-path = with pkgs; lib.makeLibraryPath [
        stdenv.cc.cc
      ];


    in
    rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python
          pre-commit
          (python3.withPackages (ps: with ps; [
            virtualenvwrapper
          ]))
        ];
        shellHook = ''
          # Augment the dynamic linker path
          export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"

          # Setup the virtual environment if it doesn't already exist.
          VENV=.venv
          if test ! -d $VENV; then
            virtualenv $VENV
            # Install Python dependencies
            pip install -r requirements_dev.txt
            pip install -e .
          fi
          source ./$VENV/bin/activate
          export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH

          pre-commit install
        '';

      };
    }
  );
}
