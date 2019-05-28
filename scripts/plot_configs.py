from matplotlib import cm

SET2COLORS = cm.get_cmap('Set2').colors
SET2 = {'darkgreen': SET2COLORS[0],
        'orange': SET2COLORS[1],
        'blue': SET2COLORS[2],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET2COLORS[6],
        'grey': SET2COLORS[7],
        }

icml_piccolo_final_configs = {
    'model-free': ('Base Algorithm',  SET2['grey']),
    'last': (r'\textsc{Last}', SET2['blue']),
    'replay': (r'\textsc{Replay}', SET2['pink']),
    'sim': (r'\textsc{TrueDyn}', SET2['lightgreen']),
    'sim0.5-VI': (r'\textsc{BiasedDyn0.5-vi}', SET2['orange']),
    'sim0.8-VI': (r'\textsc{BiasedDyn0.8-vi}', SET2['pink']),
    'last-VI': (r'\textsc{Last-vi}', SET2['orange']),
    'sim0.2-VI': (r'\textsc{BiasedDyn0.2-vi}', SET2['darkgreen']),
    'pcl-adv': (r'\textsc{PicCoLO-Adversarial}', SET2['blue']),
    'dyna-adv': (r'\textsc{DYNA-Adversarial}', SET2['pink']),
    'order': [
        'model-free', 'last', 'replay', 'sim', 'sim0.2-VI', 'sim0.5-VI', 'sim0.8-VI', 'last-VI',
        'sim0.2-VI', 'pcl-adv', 'dyna-adv']
}


class Configs(object):
    def __init__(self, style=None, colormap='Set2'):
        if not style:
            self.configs = None
            self.colors = iter(cm.get_cmap(colormap).colors)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)
