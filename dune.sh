# Check if dune_plot_style is already installed by looking for the directory named "dune_plot_style"
if [ -d "dune_plot_style" ]; then
    echo "dune_plot_style is already installed."
else
    echo "dune_plot_style not found, running dune.sh to install it." 
    export DUNE_PLOT_STYLE_LATEST_TAG=`curl --silent "https://api.github.com/repos/DUNE/dune_plot_style/releases" | jq -r 'map(select(.prerelease == false)) | first | .tag_name'`
    wget --no-check-certificate https://github.com/DUNE/dune_plot_style/archive/refs/tags/${DUNE_PLOT_STYLE_LATEST_TAG}.tar.gz -O dune_plot_style.tar.gz
    # Extract the tar.gz file and make sure to rename the directory to "dune_plot_style"
    tar -xvzf dune_plot_style.tar.gz
    export DUNE_PLOT_STYLE_VERSION=`echo ${DUNE_PLOT_STYLE_LATEST_TAG} | sed 's/^v//'`
    mv dune_plot_style-${DUNE_PLOT_STYLE_VERSION} dune_plot_style
    cd dune_plot_style
    # Install the dune_plot_style package
    python3 -m pip install .
    cd ..
fi