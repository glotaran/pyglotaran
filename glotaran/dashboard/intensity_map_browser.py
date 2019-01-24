import typing
import holoviews as hv
import xarray as xr


class IntensityMapBrowser:

    def show(self, dataset: xr.Dataset, value: str,
             height: int = 500, width: int = 500, log_scale_time=False):

        full = hv.QuadMesh(dataset[value])\
               .options(hv.opts.QuadMesh(cmap='Rainbow',
                                         colorbar=True,
                                         colorbar_position='left',
                                         width=width,
                                         height=height,
                                         logx=log_scale_time
                                         ))

        taps = hv.streams.Tap(x=0, y=0, source=full)

        def _spectrum(time, _):
            return hv.Curve(dataset[value].sel(time=time, method='nearest'))\
                .options(hv.opts.Curve(framewise=True))
        spectrum = hv.DynamicMap(_spectrum, streams=[taps])

        def sel_trace(_, spectral, log_scale_time):
            return hv.Curve(dataset[value].sel(spectral=spectral, method='nearest'))\
                .options(hv.opts.Curve(logx=log_scale_time, framewise=True))
        trace = hv.DynamicMap(sel_trace, streams=[taps])

        def marker(x, y):
            return hv.VLine(x) * hv.HLine(y)

        cross = hv.DynamicMap(marker, streams=[taps])

        return full * cross << spectrum << trace
