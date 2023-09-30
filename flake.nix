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
            pip
            pytz
            virtualenvwrapper
          ]))
        ];
        shellHook = ''
          # Augment the dynamic linker path
          export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"

          # Setup the virtual environment if it doesn't already exist.
          export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
          VENV=.venv
          if test ! -d $VENV; then
            virtualenv $VENV
            source ./$VENV/bin/activate
            # Install Python dependencies
            pip install -r requirements_dev.txt
            pip install -e .
          fi
          source ./$VENV/bin/activate

          pre-commit install
        '';

      };
    }
  );
}
