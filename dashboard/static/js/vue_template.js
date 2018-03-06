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
    modification: experiments_dir[0].modifications[0],
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
      return this.current.name + '/results/' + this.modification.name;
    },
    lossPlotPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/loss_plot.html';
    }),
    accuracyPlotPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/acc_plot.html';
    }),
    descriptionPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/description.html';
    }),
    resultMetricsPath: fullPathDecorator(function(){
      return this.current_dir + '/plots/results.html';
    }),
    schemePath: fullPathDecorator(function(){
      return this.current_dir + '/plots/scheme.png';
    })
  }
});
