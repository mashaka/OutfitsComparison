function fullPathDecorator(f) {
  return function() {
    return '../experiments/' + f.call(this, arguments);
  }
}

var app = new Vue({
  el: '#app',
  data: {
    OVERVIEW: 'Overview',
    current: experiments_dir[0],
    items: experiments_dir
  },
  methods: {
    setCurrentExperiment: function (item) {
      this.current = item;
    }
  },
  computed: {
    lossPlotPath: fullPathDecorator(function(){
      return this.current + '/plots/loss_plot.html';
    }),
    accuracyPlotPath: fullPathDecorator(function(){
      return this.current + '/plots/acc_plot.html';
    }),
    descriptionPath: fullPathDecorator(function(){
      return this.current + '/plots/description.html';
    }),
    resultMetricsPath: fullPathDecorator(function(){
      return this.current + '/plots/results.html';
    }),
    schemePath: fullPathDecorator(function(){
      return this.current + '/plots/scheme.png';
    })
  }
});
