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
    current: experiments_dir.slice(-1)[0] ,
    modification: experiments_dir.slice(-1)[0].modifications.slice(-1)[0],
    items: experiments_dir
  },
  methods: {
    setCurrentExperiment: function (item, modif) {
      this.current = item;
      this.modification = modif;
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
    })
  }
});
