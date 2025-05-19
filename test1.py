import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation

try:
    import cupy
    xp = cupy
    CUPY_AVAILABLE = True
except ImportError:
    xp = np
    CUPY_AVAILABLE = False

class IdealPluckerSurfaceVisualizer:
    def __init__(self,
                 initial_n_folds: int = 3,
                 initial_z_amplitude: float = 0.5,
                 initial_u_resolution: int = 100,
                 initial_v_resolution: int = 50,
                 initial_alpha: float = 0.85,
                 initial_azimuth: float = -60,
                 initial_elevation: float = 30,
                 initial_cmap: str = 'viridis',
                 initial_view: str = 'Surface',
                 start_animated: bool = False,
                 animation_param_name: str = 'n_folds',
                 u_range: tuple[float, float] = (0, 2 * np.pi),
                 v_range: tuple[float, float] = (-1, 1)
                 ):

        self.params = {
            'n_folds': initial_n_folds,
            'z_amplitude': initial_z_amplitude,
            'u_resolution': initial_u_resolution,
            'v_resolution': initial_v_resolution,
            'alpha': initial_alpha,
            'azimuth': initial_azimuth,
            'elevation': initial_elevation,
            'cmap': initial_cmap,
            'view_type': initial_view,
            'lighting': True,
            'edges': True
        }
        self.u_range = u_range
        self.v_range = v_range
        self.initial_params_backup = self.params.copy()

        self.fig = plt.figure(figsize=(17, 11))
        
        self.ax_main_plot = self.fig.add_axes([0.05, 0.05, 0.60, 0.9], projection='3d')

        widget_area_left = 0.68
        widget_width = 0.28
        slider_height = 0.025
        slider_spacing = 0.035
        current_y_pos = 0.92

        self.ax_widgets = {}
        
        slider_params = [
            ('n_folds', 'Folds (n)', 1, 15, 1),
            ('z_amplitude', 'Z Amplitude (c)', 0.0, 2.0, 0.05),
            ('u_resolution', 'U Resolution', 20, 400, 10),
            ('v_resolution', 'V Resolution', 10, 200, 5),
            ('alpha', 'Alpha', 0.05, 1.0, 0.05),
            ('azimuth', 'Azimuth', -180, 180, 5),
            ('elevation', 'Elevation', -90, 90, 5)
        ]

        for name, label, min_val, max_val, step in slider_params:
            self.ax_widgets[name] = self.fig.add_axes([widget_area_left, current_y_pos, widget_width, slider_height])
            current_y_pos -= slider_spacing
        
        current_y_pos -= 0.02
        self.ax_widgets['cmap'] = self.fig.add_axes([widget_area_left, current_y_pos - 0.13, widget_width * 0.45, 0.15])
        self.ax_widgets['view_type'] = self.fig.add_axes([widget_area_left + widget_width * 0.5, current_y_pos - 0.06, widget_width * 0.45, 0.08])
        self.ax_widgets['toggles'] = self.fig.add_axes([widget_area_left + widget_width * 0.5, current_y_pos - 0.13, widget_width * 0.45, 0.06])
        
        current_y_pos -= 0.18
        button_width = widget_width * 0.31
        self.ax_widgets['reset'] = self.fig.add_axes([widget_area_left, current_y_pos, button_width, 0.04])
        self.ax_widgets['save'] = self.fig.add_axes([widget_area_left + button_width + 0.01, current_y_pos, button_width, 0.04])
        self.ax_widgets['animate'] = self.fig.add_axes([widget_area_left + 2 * (button_width + 0.01), current_y_pos, button_width, 0.04])
        
        self.light = LightSource(azdeg=315, altdeg=45)
        self.surface_plot = None
        self.wireframe_plot = None
        
        m = plt.cm.ScalarMappable(cmap=plt.get_cmap(self.params['cmap']))
        m.set_array([])
        self.colorbar = self.fig.colorbar(m, ax=self.ax_main_plot, shrink=0.6, aspect=20, pad=0.05, location='left')
        
        self._create_widgets(slider_params)
        self.X, self.Y, self.Z = self._generate_data()
        self._update_plot()

        self.anim = None
        self.animate_active = False
        self.animation_param_name = animation_param_name
        if start_animated:
            self._toggle_animation(None)

    def _generate_data(self):
        u_res = int(self.params['u_resolution'])
        v_res = int(self.params['v_resolution'])

        u_vals = xp.linspace(self.u_range[0], self.u_range[1], u_res, dtype=xp.float32)
        v_vals = xp.linspace(self.v_range[0], self.v_range[1], v_res, dtype=xp.float32)
        U, V = xp.meshgrid(u_vals, v_vals)

        X_data = V * xp.cos(U)
        Y_data = V * xp.sin(U)
        Z_data = self.params['z_amplitude'] * xp.sin(int(self.params['n_folds']) * U)
        
        if CUPY_AVAILABLE:
            return cupy.asnumpy(X_data), cupy.asnumpy(Y_data), cupy.asnumpy(Z_data)
        return X_data, Y_data, Z_data

    def _setup_plot_axes(self):
        self.ax_main_plot.cla()
        self.ax_main_plot.set_xlabel('X', fontsize=10)
        self.ax_main_plot.set_ylabel('Y', fontsize=10)
        self.ax_main_plot.set_zlabel('Z', fontsize=10)
        self.ax_main_plot.tick_params(axis='both', which='major', labelsize=8)
        self.ax_main_plot.view_init(elev=self.params['elevation'], azim=self.params['azimuth'])

    def _update_plot(self, event=None):
        self._setup_plot_axes()
        self.X, self.Y, self.Z = self._generate_data()

        title_str = (f"n={int(self.params['n_folds'])}, "
                     f"c={self.params['z_amplitude']:.2f}, "
                     f"res=({int(self.params['u_resolution'])},{int(self.params['v_resolution'])})")
        self.ax_main_plot.set_title(title_str, fontsize=11, pad=15)
        
        current_cmap = plt.get_cmap(self.params['cmap'])
        edge_color = 'k' if self.params['edges'] else None
        line_width = 0.1 if self.params['edges'] else 0

        if self.params['view_type'] == 'Surface':
            if self.params['lighting']:
                rgb = self.light.shade(self.Z, cmap=current_cmap, vert_exag=0.1, blend_mode='soft')
                self.surface_plot = self.ax_main_plot.plot_surface(
                    self.X, self.Y, self.Z, facecolors=rgb,
                    edgecolor=edge_color, linewidth=line_width,
                    rstride=1, cstride=1, alpha=self.params['alpha'], antialiased=True
                )
            else:
                 self.surface_plot = self.ax_main_plot.plot_surface(
                    self.X, self.Y, self.Z, cmap=current_cmap,
                    edgecolor=edge_color, linewidth=line_width,
                    rstride=1, cstride=1, alpha=self.params['alpha'], antialiased=True
                )
            self.wireframe_plot = None
        elif self.params['view_type'] == 'Wireframe':
            self.wireframe_plot = self.ax_main_plot.plot_wireframe(
                self.X, self.Y, self.Z, color=current_cmap(0.5),
                linewidth=0.4, rstride=max(1, int(self.params['u_resolution']/75)), cstride=max(1,int(self.params['v_resolution']/35)),
                alpha=self.params['alpha']
            )
            self.surface_plot = None
        
        max_abs_xy = max(abs(self.v_range[0]), abs(self.v_range[1])) * 1.1
        max_abs_z = max(0.1, abs(self.params['z_amplitude'])) * 1.1
        
        self.ax_main_plot.set_xlim([-max_abs_xy, max_abs_xy])
        self.ax_main_plot.set_ylim([-max_abs_xy, max_abs_xy])
        self.ax_main_plot.set_zlim([-max_abs_z, max_abs_z])

        self.colorbar.mappable.set_cmap(current_cmap)
        if self.Z is not None and self.Z.size > 0:
             self.colorbar.mappable.set_clim(vmin=self.Z.min(), vmax=self.Z.max())
        
        if self.surface_plot and self.params['view_type'] == 'Surface' and not self.params['lighting']:
            self.colorbar.update_normal(self.surface_plot)
        else:
            self.colorbar.update_normal(self.colorbar.mappable)


        self.fig.canvas.draw_idle()

    def _create_widgets(self, slider_params_list):
        self.sliders = {}
        for name, label, min_val, max_val, step in slider_params_list:
            self.sliders[name] = Slider(self.ax_widgets[name], label, min_val, max_val, 
                                        valinit=self.params[name], valstep=step,
                                        valfmt=f'%0.{2 if isinstance(step, float) and step < 1 else 0}f')
            self.sliders[name].on_changed(lambda val, k=name: self._update_param_from_slider(k, val))
            self.sliders[name].label.set_fontsize(9)
            self.sliders[name].valtext.set_fontsize(9)


        cmaps_options = ['viridis', 'plasma', 'magma', 'coolwarm', 'RdYlBu', 'cividis', 'twilight_shifted', 'hsv', 'turbo']
        active_cmap_idx = cmaps_options.index(self.params['cmap']) if self.params['cmap'] in cmaps_options else 0
        self.radio_cmap = RadioButtons(self.ax_widgets['cmap'], cmaps_options, active=active_cmap_idx)
        self.radio_cmap.on_clicked(lambda label: self._update_param_from_widget('cmap', label))
        for patch in self.radio_cmap.ax.patches: # <<< ИСПРАВЛЕНИЕ ЗДЕСЬ
            if isinstance(patch, plt.Circle):
                patch.set_radius(0.035)
        for label_widget in self.radio_cmap.labels: label_widget.set_fontsize(8)


        view_options = ['Surface', 'Wireframe']
        self.radio_view_type = RadioButtons(self.ax_widgets['view_type'], view_options, active=view_options.index(self.params['view_type']))
        self.radio_view_type.on_clicked(lambda label: self._update_param_from_widget('view_type', label))
        for patch in self.radio_view_type.ax.patches: # <<< ИСПРАВЛЕНИЕ ЗДЕСЬ
            if isinstance(patch, plt.Circle):
                patch.set_radius(0.05)
        for label_widget in self.radio_view_type.labels: label_widget.set_fontsize(9)


        self.check_toggles = CheckButtons(self.ax_widgets['toggles'], ['Lighting', 'Edges'], 
                                          [self.params['lighting'], self.params['edges']])
        self.check_toggles.on_clicked(self._toggle_features)
        for label_widget in self.check_toggles.labels: label_widget.set_fontsize(9)
        for rect in self.check_toggles.rectangles: rect.set_width(0.1); rect.set_height(0.1)


        self.button_reset = Button(self.ax_widgets['reset'], 'Reset')
        self.button_reset.on_clicked(self._reset_params)
        self.button_reset.label.set_fontsize(9)

        self.button_save = Button(self.ax_widgets['save'], 'Save PNG')
        self.button_save.on_clicked(self._save_figure)
        self.button_save.label.set_fontsize(9)

        self.button_animate = Button(self.ax_widgets['animate'], 'Animate')
        self.button_animate.on_clicked(self._toggle_animation)
        self.button_animate.label.set_fontsize(9)

    def _update_param_from_slider(self, key, value):
        if key in ['n_folds', 'u_resolution', 'v_resolution']:
            self.params[key] = int(value)
        else:
            self.params[key] = value
        
        if not self.animate_active:
            self._update_plot()
            
    def _update_param_from_widget(self, key, value):
        self.params[key] = value
        if not self.animate_active:
            self._update_plot()

    def _toggle_features(self, label):
        param_key = label.lower()
        if param_key in self.params:
            self.params[param_key] = not self.params[param_key]
        
        active_states = []
        for l in self.check_toggles.labels:
            active_states.append(self.params[l.get_text().lower()])
        self.check_toggles.set_active([i for i, state in enumerate(active_states) if state])
        
        if not self.animate_active:
            self._update_plot()
            
    def _reset_params(self, event=None):
        if self.animate_active and self.anim:
            self.anim.event_source.stop()
            self.anim = None
            self.animate_active = False
            self.button_animate.label.set_text("Animate")
            self._set_widget_interaction(True)

        self.params = self.initial_params_backup.copy()
        for key, slider_widget in self.sliders.items():
            slider_widget.set_val(self.params[key])
        
        active_cmap_idx = self.radio_cmap.labels.index(next(lbl for lbl in self.radio_cmap.labels if lbl.get_text() == self.params['cmap']))
        self.radio_cmap.set_active(active_cmap_idx)
        active_view_idx = self.radio_view_type.labels.index(next(lbl for lbl in self.radio_view_type.labels if lbl.get_text() == self.params['view_type']))
        self.radio_view_type.set_active(active_view_idx)
        
        active_toggles = []
        for l in self.check_toggles.labels:
            active_toggles.append(self.params[l.get_text().lower()])
        self.check_toggles.set_active([i for i, state in enumerate(active_toggles) if state])
        
        self._update_plot()

    def _save_figure(self, event=None):
        filename = (f"plucker_n{int(self.params['n_folds'])}"
                    f"_c{self.params['z_amplitude']:.2f}"
                    f"_res{int(self.params['u_resolution'])}x{int(self.params['v_resolution'])}.png")
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='w')

    def _animation_frame(self, frame):
        if self.animation_param_name == 'n_folds':
            max_folds, min_folds = 15, 1
            val = frame % (2 * (max_folds - min_folds))
            if val > (max_folds - min_folds):
                val = 2 * (max_folds - min_folds) - val
            val += min_folds
            self.params['n_folds'] = int(val)
            self.sliders['n_folds'].set_val(int(val))
        elif self.animation_param_name == 'z_amplitude':
            min_amp, max_amp = 0.05, 1.5
            val = (frame % 100) / 100.0 * (max_amp - min_amp) + min_amp
            self.params['z_amplitude'] = val
            self.sliders['z_amplitude'].set_val(val)
        
        self._update_plot()
        return [self.ax_main_plot]

    def _set_widget_interaction(self, enabled: bool):
        for slider in self.sliders.values():
            slider.eventson = enabled
        self.radio_cmap.eventson = enabled
        self.radio_view_type.eventson = enabled
        self.check_toggles.eventson = enabled
        self.button_reset.eventson = enabled


    def _toggle_animation(self, event=None):
        if self.animate_active and self.anim:
            self.anim.event_source.stop()
            self.anim = None
            self.animate_active = False
            self.button_animate.label.set_text("Animate")
            self._set_widget_interaction(True)
        else:
            self.animate_active = True
            self.button_animate.label.set_text("Stop Anim")
            self._set_widget_interaction(False) 
            self.sliders['azimuth'].eventson = True 
            self.sliders['elevation'].eventson = True
            self.sliders['alpha'].eventson = True
            self.button_save.eventson = True


            self.anim = FuncAnimation(self.fig, self._animation_frame, frames=None,
                                      interval=70, blit=False, repeat=True, cache_frame_data=False)
            self.fig.canvas.draw_idle()
            
    def show(self):
        if CUPY_AVAILABLE:
            print("CuPy found and will be used for GPU acceleration if possible.")
        else:
            print("CuPy not found. Using NumPy for calculations (CPU).")
        plt.show()

if __name__ == "__main__":
    visualizer = IdealPluckerSurfaceVisualizer(
        initial_n_folds=2,
        initial_z_amplitude=0.4,
        initial_u_resolution=120,
        initial_v_resolution=60,
        initial_cmap='turbo',
        initial_alpha=0.9,
        start_animated=False
    )
    visualizer.show()