function fullPathDecorator(f) {
  return function() {
    return '../trained_models/' + f.call(this, arguments);
  }
}

var app = new Vue({
  el: '#app',
  data: {
    OVERVIEW: 'Overview',
    PLAN: 'Plan',
    ARTICLES: 'Articles',
    DATASETS: 'Datasets',
    MANUAL_FEATURES: 'manual_features',
    CLOTHES_COMP: 'clothes_comparation',
    METRICS: ['acc_0', 'acc_1', 'acc_2', 'precision', 'recall', 'r2', 'MAE', 'MSE', 'pairs'],
    current: 'Overview',
    modification: experiments_dir.slice(-1)[0].modifications.slice(-1)[0],
    items: experiments_dir
  },
  methods: {
    setCurrentExperiment: function (item, modif) {
      this.current = item;
      this.modification = modif;
    },
    getMetricClass: function(modif, metric) {
      if (this.bestMetricResults[metric] === modif.results[metric]) {
        return true;
      }
      return false;
    }
    
  },
  computed: {
    current_dir: function() {
      return this.current.name + '/' + this.modification.name;
    },
    lossPlotPath: fullPathDecorator(function() {
      return this.current_dir + '/plots/loss_plot.html';
    }),
    isRegression: function() {
      return this.modification.is_regression === 'True';
    },
    accuracyPlotPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/acc_plot.html';
    }),
    maePlotPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/mae_plot.html';
    }),
    msePlotPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/mse_plot.html';
    }),
    descriptionPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/description.html';
    }),
    resultMetricsPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/results.html';
    }),
    schemePath: fullPathDecorator(function(){
      return this.current_dir + '/plots/scheme.svg';
    }),
    histogramPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/histogram.html';
    }),
    bestMetricResults: function() {
      var ans = {};
      for(let metric of this.METRICS) {
        let maxValue = -20;
        for (let directory of this.items) {
          for (let modif of directory.modifications) {
            let value = modif.results[metric];
            if(metric === 'MAE' || metric === 'MSE') {
              value = -value;
            }
            if (maxValue < value) {
              maxValue = value;
            }
          }
        }
        if(metric === 'MAE' || metric === 'MSE') {
          ans[metric] = -maxValue;
        } else {
          ans[metric] = maxValue;
        }
      }
      console.log(ans)
      return ans;
    }
  }
});
